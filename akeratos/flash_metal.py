"""The Metal source code of the fused attention kernels that `akeratos.flash` compiles.

Plain strings, which `flash.py` compiles through `mx.fast.metal_kernel`. Each kernel body runs
once per thread; the template ints are BQ (query rows per threadgroup, 8 per simdgroup), BK (keys
per tile) and HD (head size).
"""

HEADER = """
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
using namespace metal;
typedef simdgroup_matrix<float, 8, 8> F8;
// The element this lane holds in an 8x8 fragment: row fm, columns fn and fn + 1 (MLX steel's
// BaseMMAFrag::get_coord). Lanes holding the same row differ in lane bits 0 and 3.
inline short frag_row(ushort lane) { return ((lane / 4) & 4) + ((lane / 2) % 4); }
inline short frag_col(ushort lane) { return ((lane / 4) & 2) * 2 + (lane % 2) * 2; }
inline float row_max(float x) {
  x = max(x, simd_shuffle_xor(x, ushort(1)));
  return max(x, simd_shuffle_xor(x, ushort(8)));
}
inline float row_sum(float x) {
  x += simd_shuffle_xor(x, ushort(1));
  return x + simd_shuffle_xor(x, ushort(8));
}
"""

# params: [B, H, KV, T]; fparams: [scale].
FWD_SRC = """
  const int H = params[1], KV = params[2], T = params[3];
  const float scale = fparams[0];
  const float cs = scale * 1.4426950408889634f;
  const int qt = int(threadgroup_position_in_grid.x);
  const int h = int(threadgroup_position_in_grid.y);
  const int b = int(threadgroup_position_in_grid.z);
  const int sg = int(simdgroup_index_in_threadgroup);
  const ushort lane = ushort(thread_index_in_simdgroup);
  const int tid = int(thread_position_in_threadgroup.x);
  constexpr int NT = (BQ / 8) * 32;
  constexpr int NJ = HD / 8;
  constexpr int NKB = BK / 8;
  const int g = h / (H / KV);
  const short fm = frag_row(lane), fn = frag_col(lane);

  threadgroup bfloat16_t Ks[BK * HD];
  threadgroup bfloat16_t Vs[BK * HD];

  const int pfirst = qt * BQ;
  const int plast = pfirst + BQ - 1;
  const int sfirst = pfirst + sg * 8;
  const int p = sfirst + fm;
  const int lo_p = lo[b * T + p];
  const int lo_s = lo[b * T + sfirst];
  const int kt0 = lo[b * T + pfirst] / BK;
  const int kt1 = plast / BK;

  const device bfloat16_t* qrow = q + ((size_t)(b * H + h) * T + p) * HD;
  F8 Qf[NJ];
  F8 Of[NJ];
  for (int j = 0; j < NJ; ++j) {
    Qf[j].thread_elements()[0] = float(qrow[j * 8 + fn]);
    Qf[j].thread_elements()[1] = float(qrow[j * 8 + fn + 1]);
    Of[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  }
  float m = -INFINITY;
  float l = 0.0f;
  const device bfloat16_t* kbase = k + (size_t)(b * KV + g) * T * HD;
  const device bfloat16_t* vbase = v + (size_t)(b * KV + g) * T * HD;

  for (int kt = kt0; kt <= kt1; ++kt) {
    const int k0 = kt * BK;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < BK * HD; i += NT) {
      Ks[i] = kbase[(size_t)k0 * HD + i];
      Vs[i] = vbase[(size_t)k0 * HD + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 > sfirst + 7 || k0 + BK - 1 < lo_s) {
      continue;  // no row of this simdgroup sees this tile (uniform per simdgroup)
    }
    F8 Sf[NKB];
    for (int kb = 0; kb < NKB; ++kb) {
      Sf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      for (int j = 0; j < NJ; ++j) {
        F8 Kt;  // K^T fragment: row = dim j*8 + fm, column = key kb*8 + fn (+1)
        Kt.thread_elements()[0] = float(Ks[(kb * 8 + fn) * HD + j * 8 + fm]);
        Kt.thread_elements()[1] = float(Ks[(kb * 8 + fn + 1) * HD + j * 8 + fm]);
        simdgroup_multiply_accumulate(Sf[kb], Qf[j], Kt, Sf[kb]);
      }
    }
    float tmax = -INFINITY;
    for (int kb = 0; kb < NKB; ++kb) {
      for (int e = 0; e < 2; ++e) {
        const int c = k0 + kb * 8 + fn + e;
        float s = Sf[kb].thread_elements()[e] * cs;
        s = (c >= lo_p && c <= p) ? s : -INFINITY;
        Sf[kb].thread_elements()[e] = s;
        tmax = max(tmax, s);
      }
    }
    tmax = row_max(tmax);
    const float mnew = max(m, tmax);
    const float muse = (mnew == -INFINITY) ? 0.0f : mnew;
    const float fac = exp2(m - muse);
    float tsum = 0.0f;
    for (int kb = 0; kb < NKB; ++kb) {
      for (int e = 0; e < 2; ++e) {
        const float pv = exp2(Sf[kb].thread_elements()[e] - muse);
        Sf[kb].thread_elements()[e] = pv;
        tsum += pv;
      }
    }
    tsum = row_sum(tsum);
    l = l * fac + tsum;
    m = mnew;
    for (int j = 0; j < NJ; ++j) {
      Of[j].thread_elements()[0] *= fac;
      Of[j].thread_elements()[1] *= fac;
      for (int kb = 0; kb < NKB; ++kb) {
        F8 Vf;  // V fragment: row = key kb*8 + fm, column = dim j*8 + fn (+1)
        Vf.thread_elements()[0] = float(Vs[(kb * 8 + fm) * HD + j * 8 + fn]);
        Vf.thread_elements()[1] = float(Vs[(kb * 8 + fm) * HD + j * 8 + fn + 1]);
        simdgroup_multiply_accumulate(Of[j], Sf[kb], Vf, Of[j]);
      }
    }
  }
  const float inv = 1.0f / l;
  device bfloat16_t* orow = o + ((size_t)(b * T + p) * H + h) * HD;
  for (int j = 0; j < NJ; ++j) {
    orow[j * 8 + fn] = static_cast<bfloat16_t>(Of[j].thread_elements()[0] * inv);
    orow[j * 8 + fn + 1] = static_cast<bfloat16_t>(Of[j].thread_elements()[1] * inv);
  }
  if (fn == 0) {
    lse[(size_t)(b * H + h) * T + p] = (m + log2(l)) * 0.6931471805599453f;
  }
"""

# dQ: one threadgroup per (query tile, head, batch); same key range as the forward pass. Each
# lane keeps its Q and dO fragment elements as bfloat16 (half the registers of float) and the
# score loop runs dim-outer, so each is converted once per key tile.
DQ_SRC = """
  const int H = params[1], KV = params[2], T = params[3];
  const float scale = fparams[0];
  const float cs = scale * 1.4426950408889634f;
  const int qt = int(threadgroup_position_in_grid.x);
  const int h = int(threadgroup_position_in_grid.y);
  const int b = int(threadgroup_position_in_grid.z);
  const int sg = int(simdgroup_index_in_threadgroup);
  const ushort lane = ushort(thread_index_in_simdgroup);
  const int tid = int(thread_position_in_threadgroup.x);
  constexpr int NT = (BQ / 8) * 32;
  constexpr int NJ = HD / 8;
  constexpr int NKB = BK / 8;
  const int g = h / (H / KV);
  const short fm = frag_row(lane), fn = frag_col(lane);

  threadgroup bfloat16_t Ks[BK * HD];
  threadgroup bfloat16_t Vs[BK * HD];

  const int pfirst = qt * BQ;
  const int plast = pfirst + BQ - 1;
  const int sfirst = pfirst + sg * 8;
  const int p = sfirst + fm;
  const int lo_p = lo[b * T + p];
  const int lo_s = lo[b * T + sfirst];
  const int kt0 = lo[b * T + pfirst] / BK;
  const int kt1 = plast / BK;
  const size_t row = (size_t)(b * H + h) * T + p;
  const float lse2 = lse[row] * 1.4426950408889634f;
  const float dvp = dvec[row];

  const device bfloat16_t* qrow = q + row * HD;
  const device bfloat16_t* dorow = dout + ((size_t)(b * T + p) * H + h) * HD;
  bfloat16_t Qb[2 * NJ];  // this lane's two elements of each Q and dO fragment, as stored
  bfloat16_t dOb[2 * NJ];
  F8 dQf[NJ];
  for (int j = 0; j < NJ; ++j) {
    Qb[2 * j] = qrow[j * 8 + fn];
    Qb[2 * j + 1] = qrow[j * 8 + fn + 1];
    dOb[2 * j] = dorow[j * 8 + fn];
    dOb[2 * j + 1] = dorow[j * 8 + fn + 1];
    dQf[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  }
  const device bfloat16_t* kbase = k + (size_t)(b * KV + g) * T * HD;
  const device bfloat16_t* vbase = v + (size_t)(b * KV + g) * T * HD;

  for (int kt = kt0; kt <= kt1; ++kt) {
    const int k0 = kt * BK;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = tid; i < BK * HD; i += NT) {
      Ks[i] = kbase[(size_t)k0 * HD + i];
      Vs[i] = vbase[(size_t)k0 * HD + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k0 > sfirst + 7 || k0 + BK - 1 < lo_s) {
      continue;
    }
    F8 Sf[NKB];
    F8 dPf[NKB];
    for (int kb = 0; kb < NKB; ++kb) {
      Sf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      dPf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    for (int j = 0; j < NJ; ++j) {
      F8 Qf;
      Qf.thread_elements()[0] = float(Qb[2 * j]);
      Qf.thread_elements()[1] = float(Qb[2 * j + 1]);
      F8 dOf;
      dOf.thread_elements()[0] = float(dOb[2 * j]);
      dOf.thread_elements()[1] = float(dOb[2 * j + 1]);
      for (int kb = 0; kb < NKB; ++kb) {
        F8 Kt;
        Kt.thread_elements()[0] = float(Ks[(kb * 8 + fn) * HD + j * 8 + fm]);
        Kt.thread_elements()[1] = float(Ks[(kb * 8 + fn + 1) * HD + j * 8 + fm]);
        simdgroup_multiply_accumulate(Sf[kb], Qf, Kt, Sf[kb]);
        F8 Vt;
        Vt.thread_elements()[0] = float(Vs[(kb * 8 + fn) * HD + j * 8 + fm]);
        Vt.thread_elements()[1] = float(Vs[(kb * 8 + fn + 1) * HD + j * 8 + fm]);
        simdgroup_multiply_accumulate(dPf[kb], dOf, Vt, dPf[kb]);
      }
    }
    for (int kb = 0; kb < NKB; ++kb) {
      for (int e = 0; e < 2; ++e) {
        const int c = k0 + kb * 8 + fn + e;
        const bool keep = c >= lo_p && c <= p;
        const float pv = keep ? exp2(Sf[kb].thread_elements()[e] * cs - lse2) : 0.0f;
        Sf[kb].thread_elements()[e] = pv * (dPf[kb].thread_elements()[e] - dvp);
      }
    }
    for (int j = 0; j < NJ; ++j) {
      for (int kb = 0; kb < NKB; ++kb) {
        F8 Kf;
        Kf.thread_elements()[0] = float(Ks[(kb * 8 + fm) * HD + j * 8 + fn]);
        Kf.thread_elements()[1] = float(Ks[(kb * 8 + fm) * HD + j * 8 + fn + 1]);
        simdgroup_multiply_accumulate(dQf[j], Sf[kb], Kf, dQf[j]);
      }
    }
  }
  device bfloat16_t* dqrow = dq + row * HD;
  for (int j = 0; j < NJ; ++j) {
    dqrow[j * 8 + fn] = static_cast<bfloat16_t>(dQf[j].thread_elements()[0] * scale);
    dqrow[j * 8 + fn + 1] = static_cast<bfloat16_t>(dQf[j].thread_elements()[1] * scale);
  }
"""

# dK and dV: one threadgroup per (key tile, key-value head, batch); BK keys (8 per simdgroup)
# stay in threadgroup memory while query tiles of BQ rows stream past, for every query head of
# the group in turn. The score loop runs dim-outer, so each K and V fragment is loaded once per
# query tile.
DKDV_SRC = """
  const int H = params[1], KV = params[2], T = params[3];
  const float scale = fparams[0];
  const float cs = scale * 1.4426950408889634f;
  const int kt = int(threadgroup_position_in_grid.x);
  const int g = int(threadgroup_position_in_grid.y);
  const int b = int(threadgroup_position_in_grid.z);
  const int sg = int(simdgroup_index_in_threadgroup);
  const ushort lane = ushort(thread_index_in_simdgroup);
  const int tid = int(thread_position_in_threadgroup.x);
  constexpr int NT = (BK / 8) * 32;
  constexpr int NJ = HD / 8;
  constexpr int NQB = BQ / 8;
  const int R = H / KV;
  const short fm = frag_row(lane), fn = frag_col(lane);

  threadgroup bfloat16_t Ks[BK * HD];
  threadgroup bfloat16_t Vs[BK * HD];
  threadgroup bfloat16_t Qs[BQ * HD];
  threadgroup bfloat16_t dOs[BQ * HD];
  threadgroup float Ls[BQ];
  threadgroup float Ds[BQ];
  threadgroup int Lo[BQ];

  const int k0 = kt * BK;
  const int klast = k0 + BK - 1;
  const int cfirst = k0 + sg * 8;
  const int c = cfirst + fm;  // this lane's key row
  const device bfloat16_t* kbase = k + ((size_t)(b * KV + g) * T + k0) * HD;
  const device bfloat16_t* vbase = v + ((size_t)(b * KV + g) * T + k0) * HD;
  for (int i = tid; i < BK * HD; i += NT) {
    Ks[i] = kbase[i];
    Vs[i] = vbase[i];
  }
  const int qt0 = k0 / BQ;
  const int qt1 = qend[b * T + klast] / BQ;
  const int kr = (sg * 8 + fm) * HD;  // this lane's row in Ks and Vs

  F8 dKf[NJ];
  F8 dVf[NJ];
  for (int j = 0; j < NJ; ++j) {
    dKf[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    dVf[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  }

  for (int r = 0; r < R; ++r) {
    const int h = g * R + r;
    for (int qt = qt0; qt <= qt1; ++qt) {
      const int q0 = qt * BQ;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const device bfloat16_t* qbase = q + ((size_t)(b * H + h) * T + q0) * HD;
      for (int i = tid; i < BQ * HD; i += NT) {
        Qs[i] = qbase[i];
        const int rr = i / HD;
        dOs[i] = dout[((size_t)(b * T + q0 + rr) * H + h) * HD + (i - rr * HD)];
      }
      for (int i = tid; i < BQ; i += NT) {
        const size_t row = (size_t)(b * H + h) * T + q0 + i;
        Ls[i] = lse[row] * 1.4426950408889634f;
        Ds[i] = dvec[row];
        Lo[i] = lo[b * T + q0 + i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (q0 + BQ - 1 < cfirst || Lo[0] > cfirst + 7) {
        continue;  // no query of this tile sees this simdgroup's keys
      }
      F8 STf[NQB];
      F8 dPTf[NQB];
      for (int qb = 0; qb < NQB; ++qb) {
        STf[qb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        dPTf[qb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      }
      for (int j = 0; j < NJ; ++j) {
        F8 Kf;  // row = key sg*8 + fm, column = dim j*8 + fn (+1)
        Kf.thread_elements()[0] = float(Ks[kr + j * 8 + fn]);
        Kf.thread_elements()[1] = float(Ks[kr + j * 8 + fn + 1]);
        F8 Vf;
        Vf.thread_elements()[0] = float(Vs[kr + j * 8 + fn]);
        Vf.thread_elements()[1] = float(Vs[kr + j * 8 + fn + 1]);
        for (int qb = 0; qb < NQB; ++qb) {
          F8 Qt;  // Q^T: row = dim j*8 + fm, column = query qb*8 + fn (+1)
          Qt.thread_elements()[0] = float(Qs[(qb * 8 + fn) * HD + j * 8 + fm]);
          Qt.thread_elements()[1] = float(Qs[(qb * 8 + fn + 1) * HD + j * 8 + fm]);
          simdgroup_multiply_accumulate(STf[qb], Kf, Qt, STf[qb]);
          F8 dOt;
          dOt.thread_elements()[0] = float(dOs[(qb * 8 + fn) * HD + j * 8 + fm]);
          dOt.thread_elements()[1] = float(dOs[(qb * 8 + fn + 1) * HD + j * 8 + fm]);
          simdgroup_multiply_accumulate(dPTf[qb], Vf, dOt, dPTf[qb]);
        }
      }
      for (int qb = 0; qb < NQB; ++qb) {
        for (int e = 0; e < 2; ++e) {
          const int qi = qb * 8 + fn + e;  // query index in the tile
          const int pq = q0 + qi;
          const bool keep = c >= Lo[qi] && c <= pq;
          const float pv = keep ? exp2(STf[qb].thread_elements()[e] * cs - Ls[qi]) : 0.0f;
          STf[qb].thread_elements()[e] = pv;
          dPTf[qb].thread_elements()[e] = pv * (dPTf[qb].thread_elements()[e] - Ds[qi]);
        }
      }
      for (int j = 0; j < NJ; ++j) {
        for (int qb = 0; qb < NQB; ++qb) {
          F8 dOf;  // row = query qb*8 + fm, column = dim j*8 + fn (+1)
          dOf.thread_elements()[0] = float(dOs[(qb * 8 + fm) * HD + j * 8 + fn]);
          dOf.thread_elements()[1] = float(dOs[(qb * 8 + fm) * HD + j * 8 + fn + 1]);
          simdgroup_multiply_accumulate(dVf[j], STf[qb], dOf, dVf[j]);
          F8 Qf;
          Qf.thread_elements()[0] = float(Qs[(qb * 8 + fm) * HD + j * 8 + fn]);
          Qf.thread_elements()[1] = float(Qs[(qb * 8 + fm) * HD + j * 8 + fn + 1]);
          simdgroup_multiply_accumulate(dKf[j], dPTf[qb], Qf, dKf[j]);
        }
      }
    }
  }
  device bfloat16_t* dkrow = dk + ((size_t)(b * KV + g) * T + c) * HD;
  device bfloat16_t* dvrow = dv + ((size_t)(b * KV + g) * T + c) * HD;
  for (int j = 0; j < NJ; ++j) {
    dkrow[j * 8 + fn] = static_cast<bfloat16_t>(dKf[j].thread_elements()[0] * scale);
    dkrow[j * 8 + fn + 1] = static_cast<bfloat16_t>(dKf[j].thread_elements()[1] * scale);
    dvrow[j * 8 + fn] = static_cast<bfloat16_t>(dVf[j].thread_elements()[0]);
    dvrow[j * 8 + fn + 1] = static_cast<bfloat16_t>(dVf[j].thread_elements()[1]);
  }
"""
