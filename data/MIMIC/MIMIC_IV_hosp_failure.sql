select
-- ids
pat.subject_id as subject_id, adm.hadm_id as hadm_id,
-- demographics
CASE WHEN pat.gender="M" THEN 1 ELSE 0 END as male,
pat.anchor_age as age,
CASE WHEN adm.race="WHITE" THEN 1 ELSE 0 END as is_white,
-- death
TIMESTAMP_DIFF(adm.dischtime, adm.admittime, HOUR) / 24 as los_hosp,
TIMESTAMP_DIFF(adm.deathtime, adm.admittime, HOUR) / 24 as time_to_death,
adm.hospital_expire_flag as event,
CASE WHEN adm.insurance='Medicare' Then 1 ELSE 0 END as ins_medicare,
CASE WHEN adm.insurance='Medicaid' Then 1 ELSE 0 END as ins_medicaid,
CASE WHEN adm.language="ENGLISH" THEN 1 ELSE 0 END as english,
CASE WHEN adm.marital_status='MARRIED' THEN 1 ELSE 0 END as marital,
CASE WHEN adm.edregtime is not null THEN 1 ELSE 0 END as had_ed,
drg.drg_severity as svrty,
drg.drg_mortality as mtlty,
from `physionet-data.mimiciv_hosp.patients` pat
inner join
`physionet-data.mimiciv_hosp.admissions` adm
on pat.subject_id=adm.subject_id
inner join
`physionet-data.mimiciv_hosp.drgcodes` drg
on
adm.subject_id=drg.subject_id
and
adm.hadm_id=drg.hadm_id
where pat.gender is not null
and adm.race is not null
and adm.race != "UNABLE TO OBTAIN"
and adm.race != "UNKNOWN"
and drg.drg_type = 'APR'