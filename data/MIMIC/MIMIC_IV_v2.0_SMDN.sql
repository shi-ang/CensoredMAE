select
-- ids
pat.subject_id as subject_id, adm.hadm_id as hadm_id, icu.stay_id as stay_id,
-- demographics
CASE WHEN pat.gender="M" THEN 1 ELSE 0 END as is_male,
CASE WHEN adm.race="WHITE" THEN 1 ELSE 0 END as is_white,
icu_detail.admission_age as age,
-- weight height
fdw.weight ,
fdh.height ,
-- LOS
icu.los as los_icu_days,
icu_detail.los_hospital as los_hosp_days,
-- death
--icu_detail.icu_intime as icu_intime,
--icu_detail.dod as dod,
TIMESTAMP_DIFF(icu_detail.dod, icu_detail.icu_intime, HOUR) / 24 as time_to_death,
case
when icu_detail.dod is null then 0
else 1
end
as death,
-- vitals labs min max mean
vitals.*,
labs.*,
sofa.*
from `physionet-data.mimiciv_hosp.patients` pat
inner join
`physionet-data.mimiciv_hosp.admissions` adm
on pat.subject_id=adm.subject_id
inner join
`physionet-data.mimiciv_icu.icustays` icu
on adm.subject_id=icu.subject_id
and
adm.hadm_id=icu.hadm_id
inner join
`physionet-data.mimiciv_derived.first_day_height` fdh
on
adm.subject_id = fdh.subject_id and icu.stay_id = fdh.stay_id
inner join
`physionet-data.mimiciv_derived.first_day_weight` fdw
on
adm.subject_id = fdw.subject_id and icu.stay_id = fdw.stay_id
inner join
`physionet-data.mimiciv_derived.icustay_detail` icu_detail
on
adm.subject_id=icu_detail.subject_id
and
adm.hadm_id=icu_detail.hadm_id
and
icu.stay_id=icu_detail.stay_id
inner join
`physionet-data.mimiciv_derived.first_day_sofa` sofa
on
adm.subject_id=sofa.subject_id
and
adm.hadm_id=sofa.hadm_id
and
icu.stay_id=sofa.stay_id
inner join
`physionet-data.mimiciv_derived.first_day_vitalsign` vitals
on
adm.subject_id=vitals.subject_id
and
icu.stay_id=vitals.stay_id
inner join
`physionet-data.mimiciv_derived.first_day_lab` labs
on
adm.subject_id=labs.subject_id
and
icu.stay_id=labs.stay_id
where icu_detail.los_icu > 1
and pat.gender is not null
and adm.race is not null
and adm.race != "UNABLE TO OBTAIN"
and adm.race != "UNKNOWN"