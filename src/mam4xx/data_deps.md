# MAM4 Wet Scavenging Inter-Level Data Dependencies


```
void compute_q_tendencies(
    const View1D &f_act_conv, const View1D &f_act_conv_coarse,
    const View1D &f_act_conv_coarse_dust, const View1D &f_act_conv_coarse_nacl,
    const View1D &scavcoefnum, const View1D &scavcoefvol, const View1D &totcond,
    const View1D &cmfdqr, const View1D &conicw, const View1D &evapc,
    const ConstView1D &evapr, const ConstView1D &prain, const ConstView1D &dlf,
    const ConstView1D &cldt, const View1D &cldcu, const View1D &cldst,
    const View1D &cldvst, const View1D &cldvcu, const View1D &sol_facti,
    const View1D &sol_factic, const View1D &sol_factb, const View1D &scavt,
    const View1D &bcscavt, const View1D &rcscavt, const View2D &rtscavt_sv,
    const View2D &state_q, const View2D &qqcw, const View2D &ptend_q,
    ConstColumnView pdel, const Real dt, const int jnummaswtr,
    const int jnv, const int mm, const int lphase, const int imode,
    const int lspec) {

  ...

  const int k_p1 = static_cast<int>(min(k + 1, nlev - 1));
  if (lphase == 1) {
    compute_q_tendencies_phase_1(
        scavt[k], bcscavt[k], rcscavt[k], rtscavt_sv_k.data(),      // output
        f_act_conv[k], scavcoefnum[k], scavcoefvol[k], totcond[k],  // input vvv
        cmfdqr[k], conicw[k], evapc[k], evapr[k], prain[k], dlf[k], cldt[k],
        cldcu[k], cldvst[k], cldvst[k_p1], cldvcu[k], cldvcu[k_p1],
        sol_facti[k], sol_factic[k], sol_factb[k], state_q(k, mm),
        ptend_q(k, mm), qqcw(k, mm), pdel[k], dt, mam_prevap_resusp_optcc,
        jnv, mm, precabs, precabc, scavabs, scavabc, precabs_base,
        precabc_base, precnums_base, precnumc_base);
  } else {
    const Real qqcw_tmp = 0.0;
    compute_q_tendencies_phase_2(
        // These are the output values
        scavt[k], bcscavt[k], rcscavt[k], rtscavt_sv_k.data(), qqcw_tmp, qqcw(k, mm), // output
        f_act_conv[k], scavcoefnum[k], scavcoefvol[k], totcond[k],                    // input vvv
        cmfdqr[k], conicw[k], evapc[k], evapr[k], prain[k], dlf[k], cldt[k],
        cldcu[k], cldvst[k], cldvst[k_p1], cldvcu[k], cldvcu[k_p1],
        sol_facti[k], sol_factic[k], sol_factb[k], pdel[k], dt,
        mam_prevap_resusp_optcc, jnv, mm, k, precabs, precabc, scavabs,
        scavabc, precabs_base, precabc_base, precnums_base, precnumc_base);
  }
}

void compute_q_tendencies_phase_1(
    Real &scavt, Real &bcscavt, Real &rcscavt, Real rtscavt_sv[],
    const Real f_act_conv, const Real scavcoefnum, const Real scavcoefvol,
    const Real totcond, const Real cmfdqr, const Real conicw, const Real evapc,
    const Real evapr, const Real prain, const Real dlf, const Real cldt,
    const Real cldcu, const Real cldvst_k, const Real cldvst_k_p1,
    const Real cldvcu_k, const Real cldvcu_k_p1, const Real sol_facti,
    const Real sol_factic, const Real sol_factb, const Real state_q,
    const Real ptend_q, const Real qqcw_sav, const Real pdel, const Real dt,
    const int mam_prevap_resusp_optcc, const int jnv, const int mm,
    Real &precabs, Real &precabc, Real &scavabs, Real &scavabc,
    Real &precabs_base, Real &precabc_base, Real &precnums_base,
    Real &precnumc_base)

    wetdep::wetdepa_v2(
        dt, pdel, cmfdqr, evapc, dlf, conicw, prain, evapr, totcond, cldt, cldcu,
        cldvcu_k, cldvcu_k_p1, cldvst_k, cldvst_k_p1, sol_factb, sol_facti,
        sol_factic, mam_prevap_resusp_optcc, is_strat_cloudborne, scavcoef,
        f_act_conv, tracer, qqcw_sav, fracis, scavt, iscavt, icscavt, isscavt,
        bcscavt, bsscavt, rcscavt, rsscavt, precabs, precabc, scavabs, scavabc,
        precabs_base, precabc_base, precnums_base, precnumc_base);
```

