import app.consts as c
from app.compute.preprocessing import *
from app.utils import get_all_data
from app.utils.show import *


def construct(ts, fhr_raw, toco_raw):
    fhr_clean, quality = clean_signal(fhr_raw, ts)
    toco_clean = clean_toco(toco_raw, ts)

    global_baseline, mask = compute_global_baseline(
        fhr_clean, ts, min_event_dur_s=15, osc_std_window_s=30
    )
    baseline, reliable = compute_baseline_line(
        fhr_clean, mask, ts, global_baseline, 200, 20
    )

    stv, ltv = variability_metrics(fhr_clean, ts)

    contractions = extract_contractions(toco_clean, ts)
    events = detect_events(
        fhr_clean,
        ts,
        baseline,
        accel_min_dur_s=5,
        decel_min_dur_s=5,
        accel_thr_bpm=10,
        decel_thr_bpm=10,
    )
    events = classify_decelerations_wrt_toco(events, toco_clean, contractions, ts)

    plot_series(ts, fhr_raw, fhr_clean, baseline)
    plot_toco(ts, toco_raw)
    plot_toco(ts, toco_clean)
    plot_variability(ts, stv, ltv)
    plot_events_on_fhr(ts, fhr_clean, events)
    plot_ctg(
        ts,
        fhr_raw,
        fhr_clean,
        toco_clean,
        baseline,
        events,
        contractions,
        stv,
        ltv,
        quality,
    )
    plot_with_mask(fhr_clean, ~mask, baseline)


if __name__ == "__main__":
    data_hypoxia = get_all_data(c.DATA_HYPOXIA_PATH)

    df = data_hypoxia["1"]
    construct(df["time_sec"].to_numpy(), df["bpm"].to_numpy(), df["uterus"].to_numpy())
