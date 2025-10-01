Excel File Analysis Report
================================================================================

Sheet: Incident
Number of columns: 72
Columns:
   1. incident_id
   2. occurrence_date
   3. incident_type
   4. section
   5. title
   6. status
   7. category
   8. description
   9. company
  10. location
  11. sublocation
  12. department
  13. sub_department
  14. location.1
  15. repeated_incident
  16. dlevel_committee_number_epcl
  17. incident_time
  18. person_involved
  19. relationship_of_person_involved
  20. date_shift_began
  21. time_shift_began
  22. reported_by
  23. reported_date
  24. entered_date
  25. responsible_for_hse_quality_check
  26. responsible_for_investigation
  27. target_completion_date
  28. sequence_of_events
  29. quantity_contaminated
  30. why_1
  31. answer_1
  32. why_2
  33. answer_2
  34. why_3
  35. answer_3
  36. management_system_noncompliance
  37. psm
  38. conclusion
  39. completion_date
  40. type_of_equipment_failure
  41. equipment
  42. is_the_equipment_safety_critical
  43. equipment_id
  44. pse_category
  45. tier_3_description
  46. material_involved
  47. hse_site_rules_category
  48. relevant_consequence_incident
  49. worst_case_consequence_incident
  50. actual_consequence_incident
  51. root_cause
  52. responsible_for_investigation_approval
  53. investigation_type
  54. pse_category_selection_criteria
  55. actual_leakage_quantity
  56. corrective_actions
  57. entered_investigation
  58. entered_pending_closure
  59. entered_closed
  60. task_or_activity_at_time_of_incident1
  61. key_factor
  62. contributing_factor
  63. root_cause_is_missing
  64. corrective_actions_is_missing
  65. severity_score
  66. reporting_delay_days
  67. resolution_time_days
  68. estimated_cost_impact
  69. estimated_manhours_impact
  70. risk_score
  71. department_avg_risk
  72. compliance_systems_involved

First 5 rows of data:
       incident_id occurrence_date               incident_type    section              title   status       category                    description                      company       location             sublocation           department sub_department location.1 repeated_incident dlevel_committee_number_epcl  incident_time  person_involved relationship_of_person_involved  date_shift_began  time_shift_began                  reported_by reported_date entered_date responsible_for_hse_quality_check responsible_for_investigation target_completion_date             sequence_of_events quantity_contaminated  why_1  answer_1  why_2  answer_2  why_3  answer_3 management_system_noncompliance                   psm                     conclusion completion_date  type_of_equipment_failure  equipment  is_the_equipment_safety_critical  equipment_id pse_category  tier_3_description material_involved  hse_site_rules_category relevant_consequence_incident worst_case_consequence_incident actual_consequence_incident                     root_cause responsible_for_investigation_approval  investigation_type  pse_category_selection_criteria actual_leakage_quantity             corrective_actions entered_investigation entered_pending_closure       entered_closed  task_or_activity_at_time_of_incident1                     key_factor            contributing_factor  root_cause_is_missing  corrective_actions_is_missing  severity_score  reporting_delay_days  resolution_time_days  estimated_cost_impact  estimated_manhours_impact  risk_score  department_avg_risk  compliance_systems_involved
0  IN-20220405-001      2022-04-05  Other; No Loss / No Injury  Technical  OVR catalyst loss   Closed       Incident  High catalyst loss has bee...  Engro Polymer and Chemicals        Karachi  Manufacturing Facility  Process - EDC / VCM      EDC / VCM   EVCM 200               Yes      22, Process Engineering            NaN              NaN                       Employee                NaN               NaN  Kaiwan Shahzad - EPCL Plant    2023-10-03   2023-10-03  Safeer Hussain - ECORP KHI HO      Kaiwan Shahzad - EPCL Plant             2023-07-20  Cu (ppm) recorded increasi...                   SCF    NaN       NaN    NaN       NaN    NaN       NaN                            PSM   Mechanical Integrity  High catalyst loss has bee...      2023-10-05                        NaN        NaN                            NaN              NaN       Tier 1                 NaN          Catalyst                      NaN                         Asset                    C3 - Severe           C0 - No Ill Effect  Inadequate preventive/corr...      Tauqir Nasir - EPCL Plant          Team Investigation                            NaN                       SCF            AC-INC-20231005-019   2023-10-04 00:00:00     2023-10-05 00:00:00  2023-10-09 00:00:00                            NaN          Repeat failure of facility...  Repeat failure of facility...                  False                          False               3                   546                   2.0                  75000                        160         4.5             3.158333                            1
1  IN-20220405-001             NaT                         NaN        NaN  OVR catalyst loss  Unknown  Not Specified  High catalyst loss has bee...                          NaN  Not Specified                     NaN         Not Assigned            NaN        NaN               NaN                          NaN            NaN              NaN                            NaN                NaN               NaN                          NaN           NaT          NaT                            NaN                              NaN                    NaT                            NaN                   NaN    NaN       NaN    NaN       NaN    NaN       NaN                            NaN                    NaN                            NaN             NaT                        NaN        NaN                            NaN              NaN          NaN                 NaN               NaN                      NaN                           NaN                            NaN                          NaN                            NaN                            NaN                         NaN                            NaN                       NaN  OVR Dip pipe metallurgy.St...                   NaN                     NaN                  NaN                            NaN                                    NaN                            NaN                   True                          False               1                     0                   NaN                  10000                         40         1.0             1.002688                            0
2  IN-20220405-001             NaT                         NaN        NaN  OVR catalyst loss  Unknown  Not Specified  High catalyst loss has bee...                          NaN  Not Specified                     NaN         Not Assigned            NaN        NaN               NaN                          NaN            NaN              NaN                            NaN                NaN               NaN                          NaN           NaT          NaT                            NaN                              NaN                    NaT                            NaN                   NaN    NaN       NaN    NaN       NaN    NaN       NaN                            NaN                    NaN                            NaN             NaT                        NaN        NaN                            NaN              NaN          NaN                 NaN               NaN                      NaN                           NaN                            NaN                          NaN                            NaN                            NaN                         NaN                            NaN                       NaN                            NaN                   NaN                     NaN                  NaN                            NaN                                    NaN                            NaN                   True                           True               1                     0                   NaN                  10000                         40         1.0             1.002688                            0
3  IN-20220405-001             NaT                         NaN        NaN  OVR catalyst loss  Unknown  Not Specified  High catalyst loss has bee...                          NaN  Not Specified                     NaN         Not Assigned            NaN        NaN               NaN                          NaN            NaN              NaN                            NaN                NaN               NaN                          NaN           NaT          NaT                            NaN                              NaN                    NaT                            NaN                   NaN    NaN       NaN    NaN       NaN    NaN       NaN                            NaN                    NaN                            NaN             NaT                        NaN        NaN                            NaN              NaN          NaN                 NaN               NaN                      NaN                           NaN                            NaN                          NaN                            NaN                            NaN                         NaN                            NaN                       NaN          ; AC-INC-20231005-007                   NaN                     NaN                  NaN                            NaN                                    NaN                            NaN                   True                          False               1                     0                   NaN                  10000                         40         1.0             1.002688                            0
4  IN-20220405-001             NaT                         NaN        NaN  OVR catalyst loss  Unknown  Not Specified  High catalyst loss has bee...                          NaN  Not Specified                     NaN         Not Assigned            NaN        NaN               NaN                          NaN            NaN              NaN                            NaN                NaN               NaN                          NaN           NaT          NaT                            NaN                              NaN                    NaT                            NaN                   NaN    NaN       NaN    NaN       NaN    NaN       NaN                            NaN                    NaN                            NaN             NaT                        NaN        NaN                            NaN              NaN          NaN                 NaN               NaN                      NaN                           NaN                            NaN                          NaN                            NaN                            NaN                         NaN                            NaN                       NaN  OVR diffuser cup metallurg...                   NaN                     NaN                  NaN                            NaN                                    NaN                            NaN                   True                          False               1                     0                   NaN                  10000                         40         1.0             1.002688                            0

================================================================================

Sheet: Hazard ID
Number of columns: 35
Columns:
   1. incident_id
   2. occurrence_date
   3. incident_type
   4. section
   5. title
   6. status
   7. category
   8. description
   9. company
  10. location
  11. sublocation
  12. department
  13. sub_department
  14. location.1
  15. dlevel_committee_number_epcl
  16. incident_time
  17. repeated_event
  18. reported_by
  19. reported_date
  20. entered_date
  21. relevant_consequence_hazard_id
  22. worst_case_consequence_potential_hazard_id
  23. violation_type_hazard_id
  24. responsible_for_approval_hazard_id
  25. corrective_actions
  26. entered_review
  27. entered_closed
  28. corrective_actions_is_missing
  29. severity_score
  30. reporting_delay_days
  31. estimated_cost_impact
  32. estimated_manhours_impact
  33. risk_score
  34. department_avg_risk
  35. compliance_systems_involved

First 5 rows of data:
       incident_id occurrence_date                  incident_type     section                          title  status   category                    description                      company location             sublocation department      sub_department         location.1   dlevel_committee_number_epcl  incident_time repeated_event             reported_by reported_date entered_date relevant_consequence_hazard_id worst_case_consequence_potential_hazard_id violation_type_hazard_id responsible_for_approval_hazard_id  corrective_actions       entered_review       entered_closed  corrective_actions_is_missing  severity_score  reporting_delay_days  estimated_cost_impact  estimated_manhours_impact  risk_score  department_avg_risk  compliance_systems_involved
0  HA-20230316-009      2023-03-16            No Loss / No Injury    Projects  One Desiccator fell down f...  Closed  Hazard ID  Around 1235 hrs. at Produc...  Engro Polymer and Chemicals  Karachi         Projects and BD        HPO   HPO Contstruction                HPO  190, Mechanical (Projects)...            NaN             No  Majid Khan - EPCL P&BD    2023-03-20   2023-03-20                         People                     C1 - Minor                           Unsafe Act  Muhammad Umar Shafiq - EPC...                     NaN  2023-03-20 00:00:00  2023-04-05 00:00:00                           True               1                     4                   1000                         16           1             1.822430                            0
1  HA-20230318-004      2023-03-18  Process Safety Event; No L...  Production  HP Jetting Pump PU-3105 re...  Closed  Hazard ID  HP Jetting Pump PU-3105 re...  Engro Polymer and Chemicals  Karachi  Manufacturing Facility        PVC             PVC III  PVC III Feedstock                 Not Applicable            NaN             No  Yasir Ali - EPCL Plant    2023-04-03   2023-04-03                         People                   C2 - Serious                     Unsafe Condition        Amer Ahmed - EPCL Plant                     NaN  2023-04-03 00:00:00  2023-04-27 00:00:00                           True               2                    16                   2000                         24           2             0.847222                            0
2  HA-20230321-001      2023-03-21            No Loss / No Injury    Projects  At pipe rack area EEL subc...  Closed  Hazard ID  On 3/21/2023, At pipe rack...  Engro Polymer and Chemicals  Karachi         Projects and BD        HPO   HPO Contstruction   HPO Process Area  190, Mechanical (Projects)...            NaN             No  Majid Khan - EPCL P&BD    2023-03-23   2023-03-23                         People                     C1 - Minor                Safety Rule Violation  Muhammad Umar Shafiq - EPC...                     NaN  2023-03-23 00:00:00  2023-03-31 00:00:00                           True               1                     2                   1000                         16           1             1.822430                            0
3  HA-20230321-002      2023-03-21            No Loss / No Injury    Projects  EPIC fabricator was using ...  Closed  Hazard ID  Self-reporting: On 21st Ma...  Engro Polymer and Chemicals  Karachi         Projects and BD       HTDC  HTDC Contstruction               HTDC  190, Mechanical (Projects)...            NaN             No  Majid Khan - EPCL P&BD    2023-03-23   2023-03-23                         People                   C2 - Serious                           Unsafe Act        Ammar Ahmad - EPCL P&BD                     NaN  2023-03-23 00:00:00  2023-04-04 00:00:00                           True               2                     2                   2000                         24           2             1.857143                            0
4  HA-20230322-003      2023-03-22                          Other    Projects  EEL admin team members wer...  Closed  Hazard ID  On 22nd Mar 2023, At EEL o...  Engro Polymer and Chemicals  Karachi         Projects and BD        HPO   HPO Contstruction  Container Offices  190, Mechanical (Projects)...            NaN             No  Majid Khan - EPCL P&BD    2023-03-23   2023-03-23                         People                   C2 - Serious                Safety Rule Violation  Muhammad Umar Shafiq - EPC...                     NaN  2023-03-23 00:00:00  2023-03-31 00:00:00                           True               2                     1                   2000                         24           2             1.822430                            0

================================================================================

Sheet: Audit
Number of columns: 36
Columns:
   1. audit_id
   2. audit_location
   3. audit_title
   4. auditor
   5. start_date
   6. audit_status
   7. audit_category
   8. auditing_body
   9. audit_rating
  10. company
  11. location
  12. audit_type_epcl
  13. template
  14. template_version
  15. created_by
  16. audit_team
  17. responsible_for_action_plan
  18. responsible_for_action_plan_review
  19. entered_scheduled
  20. entered_in_progress
  21. entered_review
  22. entered_pending_action_plan
  23. entered_review_action_plan
  24. entered_pending_closure
  25. entered_closed
  26. checklist_category
  27. question
  28. regulatory_reference
  29. help_text
  30. answer
  31. recommendation
  32. response
  33. finding
  34. finding_location
  35. worst_case_consequence
  36. compliance_systems_involved

First 5 rows of data:
          audit_id               audit_location                    audit_title                        auditor start_date audit_status audit_category auditing_body  audit_rating company                 location                audit_type_epcl                  template template_version                     created_by                     audit_team    responsible_for_action_plan responsible_for_action_plan_review entered_scheduled entered_in_progress entered_review entered_pending_action_plan entered_review_action_plan entered_pending_closure entered_closed        checklist_category    question  regulatory_reference                      help_text                  answer  recommendation  response                        finding               finding_location         worst_case_consequence  compliance_systems_involved
0  AU-20230101-001                           CA  Hydrogen vent stack falme ...    Ghulam Murtaza - EPCL Plant 2023-01-01       Closed          Audit          Self           NaN    EPCL  CA-1650 and HCL Loading             119-Internal Audit  General Audit Check List  Current Version    Ghulam Murtaza - EPCL Plant        Moin Nasir - EPCL Plant  Ahtisham Qadir Malik - EPC...  Ahtisham Qadir Malik - EPC...            2023-09-27          2023-09-27     2023-09-27         2023-09-27 00:00:00        2023-09-29 00:00:00     2023-09-29 00:00:00     2023-09-30  General Audit Check List  Audit Type                   NaN  Just add audit type here a...                     NaN             NaN       NaN                            NaN                            NaN                            NaN                            0
1  AU-20230130-001       Manufacturing Facility          Marsh Insurance Audit  Ahtisham Qadir Malik - EPC... 2023-01-30       Closed          Audit     3rd Party           NaN    EPCL           Admin Building             16-Insurance Audit  General Audit Check List  Current Version  Safeer Hussain - ECORP KHI HO  Junaid Rafey - EPCL Plant;...  Safeer Hussain - ECORP KHI HO  Ahtisham Qadir Malik - EPC...            2023-05-25          2023-05-25     2023-05-29         2023-12-31 00:00:00        2024-04-08 00:00:00     2024-05-11 00:00:00     2025-05-31  General Audit Check List  Audit Type                   NaN  Just add audit type here a...  Insurance Survey Audit             NaN       NaN  It was observed that maint...  Maintenance; Health, Safet...  C2 - Serious; C3 - Severe;...                            0
2  AU-20230131-002              Asset Integrity  Marsh Audit 2023 - Inspect...      Junaid Rafey - EPCL Plant 2023-01-31       Closed          Audit     3rd Party           NaN    EPCL           Admin Building             16-Insurance Audit  General Audit Check List  Current Version   Ghulam Muzammil - EPCL Plant  Muhammad Shoquaib Farooq -...   Ghulam Muzammil - EPCL Plant  Safeer Hussain - ECORP KHI HO                   NaT          2023-10-24     2023-10-24         2023-10-24 00:00:00        2023-10-24 00:00:00     2023-10-24 00:00:00     2023-12-30  General Audit Check List  Audit Type                   NaN  Just add audit type here a...                     NaN             NaN       NaN  23.01 Management of piping...  Asset Integrity; Asset Int...                              ;                            0
3  AU-20230301-004  Engro Polymer and Chemicals  RMA Audit: Piping, Piping ...  Syed Tauseef Ali - EPCL Plant 2023-03-01       Closed          Audit     1st Party  Satisfactory    EPCL                 EVCM 300  82-Piping, Piping supports...  General Audit Check List  Current Version  Syed Tauseef Ali - EPCL Plant      Junaid Rafey - EPCL Plant      Junaid Rafey - EPCL Plant      Junaid Rafey - EPCL Plant            2023-03-29          2023-03-29     2023-03-29         2023-03-29 00:00:00        2023-03-29 00:00:00     2023-03-29 00:00:00     2023-03-31  General Audit Check List  Audit Type                   NaN  Just add audit type here a...                     NaN             NaN       NaN            Refer attached file    Engro Polymer and Chemicals                            NaN                            0
4  AU-20230301-005  Engro Polymer and Chemicals  RMA Audit: Temporary repai...  Syed Tauseef Ali - EPCL Plant 2023-03-01       Closed          Audit     1st Party  Satisfactory    EPCL                 EVCM 300  74-Temporary Repair Field ...  General Audit Check List  Current Version  Syed Tauseef Ali - EPCL Plant  Junaid Rafey - EPCL Plant;...      Junaid Rafey - EPCL Plant      Junaid Rafey - EPCL Plant            2023-03-29          2023-03-29     2023-03-29         2023-03-29 00:00:00        2023-03-29 00:00:00     2023-03-29 00:00:00     2023-03-31  General Audit Check List  Audit Type                   NaN  Just add audit type here a...                     NaN             NaN       NaN  Refer attached file; Refer...  Engro Polymer and Chemical...         C1 - Minor; C1 - Minor                            0

================================================================================

Sheet: Audit Findings
Number of columns: 34
Columns:
   1. audit_id
   2. audit_location
   3. audit_title
   4. auditor
   5. start_date
   6. audit_status
   7. audit_category
   8. auditing_body
   9. audit_rating
  10. company
  11. location
  12. audit_type_epcl
  13. template
  14. template_version
  15. created_by
  16. audit_team
  17. responsible_for_action_plan
  18. responsible_for_action_plan_review
  19. entered_scheduled
  20. entered_in_progress
  21. entered_review
  22. entered_pending_action_plan
  23. entered_review_action_plan
  24. entered_pending_closure
  25. entered_closed
  26. checklist_category
  27. question
  28. help_text
  29. answer
  30. recommendation
  31. finding
  32. finding_location
  33. worst_case_consequence
  34. compliance_systems_involved

First 5 rows of data:
          audit_id          audit_location            audit_title                        auditor start_date audit_status audit_category auditing_body  audit_rating company        location     audit_type_epcl                  template template_version                     created_by                     audit_team    responsible_for_action_plan responsible_for_action_plan_review entered_scheduled entered_in_progress entered_review entered_pending_action_plan entered_review_action_plan entered_pending_closure entered_closed  checklist_category  question  help_text  answer  recommendation                        finding               finding_location worst_case_consequence  compliance_systems_involved
0  AU-20230130-001  Manufacturing Facility  Marsh Insurance Audit  Ahtisham Qadir Malik - EPC... 2023-01-30       Closed          Audit     3rd Party           NaN    EPCL  Admin Building  16-Insurance Audit  General Audit Check List  Current Version  Safeer Hussain - ECORP KHI HO  Junaid Rafey - EPCL Plant;...  Safeer Hussain - ECORP KHI HO  Ahtisham Qadir Malik - EPC...            2023-05-25          2023-05-25     2023-05-29         2023-12-31 00:00:00        2024-04-08 00:00:00     2024-05-11 00:00:00     2025-05-31                 NaN       NaN        NaN     NaN             NaN  It was observed that permi...  Health, Safety and Environ...            C3 - Severe                            0
1  AU-20230130-001  Manufacturing Facility  Marsh Insurance Audit  Ahtisham Qadir Malik - EPC... 2023-01-30       Closed          Audit     3rd Party           NaN    EPCL  Admin Building  16-Insurance Audit  General Audit Check List  Current Version  Safeer Hussain - ECORP KHI HO  Junaid Rafey - EPCL Plant;...  Safeer Hussain - ECORP KHI HO  Ahtisham Qadir Malik - EPC...            2023-05-25          2023-05-25     2023-05-29         2023-12-31 00:00:00        2024-04-08 00:00:00     2024-05-11 00:00:00     2025-05-31                 NaN       NaN        NaN     NaN             NaN  It was observed that on th...  Health, Safety and Environ...            C3 - Severe                            0
2  AU-20230130-001  Manufacturing Facility  Marsh Insurance Audit  Ahtisham Qadir Malik - EPC... 2023-01-30       Closed          Audit     3rd Party           NaN    EPCL  Admin Building  16-Insurance Audit  General Audit Check List  Current Version  Safeer Hussain - ECORP KHI HO  Junaid Rafey - EPCL Plant;...  Safeer Hussain - ECORP KHI HO  Ahtisham Qadir Malik - EPC...            2023-05-25          2023-05-25     2023-05-29         2023-12-31 00:00:00        2024-04-08 00:00:00     2024-05-11 00:00:00     2025-05-31                 NaN       NaN        NaN     NaN             NaN  It was observed that, the ...  Health, Safety and Environ...            C3 - Severe                            0
3  AU-20230130-001  Manufacturing Facility  Marsh Insurance Audit  Ahtisham Qadir Malik - EPC... 2023-01-30       Closed          Audit     3rd Party           NaN    EPCL  Admin Building  16-Insurance Audit  General Audit Check List  Current Version  Safeer Hussain - ECORP KHI HO  Junaid Rafey - EPCL Plant;...  Safeer Hussain - ECORP KHI HO  Ahtisham Qadir Malik - EPC...            2023-05-25          2023-05-25     2023-05-29         2023-12-31 00:00:00        2024-04-08 00:00:00     2024-05-11 00:00:00     2025-05-31                 NaN       NaN        NaN     NaN             NaN  It was observed that maint...                    Maintenance           C2 - Serious                            0
4  AU-20230130-001  Manufacturing Facility  Marsh Insurance Audit  Ahtisham Qadir Malik - EPC... 2023-01-30       Closed          Audit     3rd Party           NaN    EPCL  Admin Building  16-Insurance Audit  General Audit Check List  Current Version  Safeer Hussain - ECORP KHI HO  Junaid Rafey - EPCL Plant;...  Safeer Hussain - ECORP KHI HO  Ahtisham Qadir Malik - EPC...            2023-05-25          2023-05-25     2023-05-29         2023-12-31 00:00:00        2024-04-08 00:00:00     2024-05-11 00:00:00     2025-05-31                 NaN       NaN        NaN     NaN             NaN  It was observed that there...           Process - UTY and PP           C2 - Serious                            0

================================================================================

Sheet: Inspection
Number of columns: 41
Columns:
   1. audit_id
   2. audit_location
   3. audit_title
   4. auditor
   5. start_date
   6. audit_status
   7. audit_category
   8. auditing_body
   9. company
  10. location
  11. audit_type_epcl
  12. template
  13. template_version
  14. created_by
  15. audit_team
  16. entered_scheduled
  17. entered_in_progress
  18. entered_review
  19. entered_closed
  20. checklist_category
  21. question
  22. regulatory_reference
  23. help_text
  24. answer
  25. recommendation
  26. response
  27. finding
  28. finding_location
  29. worst_case_consequence
  30. action_item_number
  31. action_item_title
  32. action_item_description
  33. action_item_responsible
  34. action_item_responsible_for_verification
  35. action_item_effective
  36. action_item_verification_details
  37. action_item_priority
  38. action_item_due_date
  39. action_item_status
  40. action_item_progress_notes
  41. compliance_systems_involved

First 5 rows of data:
          audit_id      audit_location                    audit_title                      auditor start_date audit_status audit_category auditing_body company         location audit_type_epcl                       template template_version                   created_by                audit_team  entered_scheduled  entered_in_progress       entered_review entered_closed             checklist_category                       question           regulatory_reference      help_text                         answer recommendation       response                        finding               finding_location worst_case_consequence  action_item_number  action_item_title  action_item_description  action_item_responsible  action_item_responsible_for_verification  action_item_effective  action_item_verification_details  action_item_priority  action_item_due_date  action_item_status  action_item_progress_notes  compliance_systems_involved
0  IS-20230904-011  HTDC Contstruction  Management Safety Audit (MSA)    Rafay Shahid - EPCL Plant 2023-09-04       Closed     Inspection          Self    EPCL             HTDC           6-MSA  Management Safety Audit (MSA)  Current Version    Rafay Shahid - EPCL Plant                       NaN                NaN  2023-09-11 00:00:00  2023-09-11 00:00:00     2024-01-29  Conversation with Workforc...  The conversation took plac...  ; ; Unsafe Act is defined ...  ; ; ; ; ; ; ;  Yes; Yes; Yes; 8 - Positio...  ; ; ; ; ; ; ;  ; ; ; ; ; ; ;  Cables were lying haphazar...  HTDC Contstruction; HTDC C...                      ;                 NaN                NaN                      NaN                      NaN                            NaN                               NaN                            NaN                      NaN                   NaN                 NaN                         NaN                            0
1  IS-20230904-013            Projects  Management Safety Audit (MSA)   Arsslan Bhatti - EPCL P&BD 2023-09-04  In Progress     Inspection          Self    EPCL             HTDC           6-MSA  Management Safety Audit (MSA)  Current Version   Arsslan Bhatti - EPCL P&BD                       NaN                NaN  2023-09-30 00:00:00                  NaN            NaT  Conversation with Workforc...  The conversation took plac...  ; ; Unsafe Act is defined ...  ; ; ; ; ; ; ;  Yes; PPE compliance; Yes; ...  ; ; ; ; ; ; ;  ; ; ; ; ; ; ;                            NaN                            NaN                    NaN                 NaN                NaN                      NaN                      NaN                            NaN                               NaN                            NaN                      NaN                   NaN                 NaN                         NaN                            0
2  IS-20230904-014            Projects  Management Safety Audit (MSA)   Arsslan Bhatti - EPCL P&BD 2023-09-04  In Progress     Inspection          Self    EPCL             HTDC           6-MSA  Management Safety Audit (MSA)  Current Version   Arsslan Bhatti - EPCL P&BD                       NaN                NaN  2023-09-30 00:00:00                  NaN            NaT  Conversation with Workforc...  The conversation took plac...  ; ; Unsafe Act is defined ...  ; ; ; ; ; ; ;  Yes; PPE compliance; Yes; ...  ; ; ; ; ; ; ;  ; ; ; ; ; ; ;                            NaN                            NaN                    NaN                 NaN                NaN                      NaN                      NaN                            NaN                               NaN                            NaN                      NaN                   NaN                 NaN                         NaN                            0
3  IS-20230904-015            Projects  Management Safety Audit (MSA)   Arsslan Bhatti - EPCL P&BD 2023-09-04       Closed     Inspection          Self    EPCL             HTDC           6-MSA  Management Safety Audit (MSA)  Current Version   Arsslan Bhatti - EPCL P&BD                       NaN                NaN  2023-09-30 00:00:00  2023-09-30 00:00:00     2024-01-29  Conversation with Workforc...  The conversation took plac...  ; ; Unsafe Act is defined ...  ; ; ; ; ; ; ;  Yes; PPE compliance; Yes; ...  ; ; ; ; ; ; ;  ; ; ; ; ; ; ;                            NaN                            NaN                    NaN                 NaN                NaN                      NaN                      NaN                            NaN                               NaN                            NaN                      NaN                   NaN                 NaN                         NaN                            0
4  IS-20230904-016    Stationary - PVC  Management Safety Audit (MSA)  Ahrad Bin Riaz - EPCL Plant 2023-09-04       Closed     Inspection          Self    EPCL  PVC I Front End           6-MSA  Management Safety Audit (MSA)  Current Version  Ahrad Bin Riaz - EPCL Plant  Zafar Tariq - EPCL Plant                NaN  2023-10-01 00:00:00  2023-10-01 00:00:00     2023-10-31  Conversation with Workforc...  The conversation took plac...  ; ; Unsafe Act is defined ...  ; ; ; ; ; ; ;  Yes; PPEs Compliance; No; ...  ; ; ; ; ; ; ;  ; ; ; ; ; ; ;  No any major damage / PPEs...               Stationary - PVC                    NaN                 NaN                NaN                      NaN                      NaN                            NaN                               NaN                            NaN                      NaN                   NaN                 NaN                         NaN                            0

================================================================================

Sheet: Inspection Findings
Number of columns: 37
Columns:
   1. audit_id
   2. audit_location
   3. audit_title
   4. auditor
   5. start_date
   6. audit_status
   7. audit_category
   8. auditing_body
   9. company
  10. location
  11. audit_type_epcl
  12. template
  13. template_version
  14. created_by
  15. audit_team
  16. entered_scheduled
  17. entered_in_progress
  18. entered_review
  19. entered_closed
  20. checklist_category
  21. question
  22. regulatory_reference
  23. answer
  24. finding
  25. finding_location
  26. action_item_number
  27. action_item_title
  28. action_item_description
  29. action_item_responsible
  30. action_item_responsible_for_verification
  31. action_item_effective
  32. action_item_verification_details
  33. action_item_priority
  34. action_item_due_date
  35. action_item_status
  36. action_item_progress_notes
  37. compliance_systems_involved

First 5 rows of data:
          audit_id      audit_location                    audit_title                      auditor start_date audit_status audit_category auditing_body company                    location audit_type_epcl                       template template_version                   created_by                audit_team  entered_scheduled  entered_in_progress       entered_review entered_closed             checklist_category                       question           regulatory_reference                answer                        finding    finding_location  action_item_number  action_item_title  action_item_description  action_item_responsible  action_item_responsible_for_verification  action_item_effective  action_item_verification_details  action_item_priority  action_item_due_date  action_item_status  action_item_progress_notes  compliance_systems_involved
0  IS-20230904-011  HTDC Contstruction  Management Safety Audit (MSA)    Rafay Shahid - EPCL Plant 2023-09-04       Closed     Inspection          Self    EPCL                        HTDC           6-MSA  Management Safety Audit (MSA)  Current Version    Rafay Shahid - EPCL Plant                       NaN                NaN  2023-09-11 00:00:00  2023-09-11 00:00:00     2024-01-29                 Audit Findings          Was it an Unsafe Act?  Unsafe Act is defined as F...                   Yes  Cables were lying haphazar...  HTDC Contstruction                 NaN                NaN                      NaN                      NaN                            NaN                               NaN                            NaN                      NaN                   NaN                 NaN                         NaN                            0
1  IS-20230904-011  HTDC Contstruction  Management Safety Audit (MSA)    Rafay Shahid - EPCL Plant 2023-09-04       Closed     Inspection          Self    EPCL                        HTDC           6-MSA  Management Safety Audit (MSA)  Current Version    Rafay Shahid - EPCL Plant                       NaN                NaN  2023-09-11 00:00:00  2023-09-11 00:00:00     2024-01-29                  Key Strengths                Key Strength(s)                            NaN                   Yes  Area housekeeping was foun...  HTDC Contstruction                 NaN                NaN                      NaN                      NaN                            NaN                               NaN                            NaN                      NaN                   NaN                 NaN                         NaN                            0
2  IS-20230904-016    Stationary - PVC  Management Safety Audit (MSA)  Ahrad Bin Riaz - EPCL Plant 2023-09-04       Closed     Inspection          Self    EPCL             PVC I Front End           6-MSA  Management Safety Audit (MSA)  Current Version  Ahrad Bin Riaz - EPCL Plant  Zafar Tariq - EPCL Plant                NaN  2023-10-01 00:00:00  2023-10-01 00:00:00     2023-10-31                  Key Strengths                Key Strength(s)                            NaN       PPEs Compliance  No any major damage / PPEs...    Stationary - PVC                 NaN                NaN                      NaN                      NaN                            NaN                               NaN                            NaN                      NaN                   NaN                 NaN                         NaN                            0
3  IS-20230904-017                 HPO  Management Safety Audit (MSA)  Hassan Masroor - EPCL Plant 2023-09-04       Review     Inspection          Self    EPCL  HPO Product Tank farm Area           6-MSA  Management Safety Audit (MSA)  Current Version  Hassan Masroor - EPCL Plant                       NaN                NaN  2023-10-02 00:00:00  2023-10-02 00:00:00            NaT                 Audit Findings          Was it an Unsafe Act?  Unsafe Act is defined as F...                   Yes  Asif MW was working on cle...                 HPO                 NaN                NaN                      NaN                      NaN                            NaN                               NaN                            NaN                      NaN                   NaN                 NaN                         NaN                            0
4  IS-20230904-017                 HPO  Management Safety Audit (MSA)  Hassan Masroor - EPCL Plant 2023-09-04       Review     Inspection          Self    EPCL  HPO Product Tank farm Area           6-MSA  Management Safety Audit (MSA)  Current Version  Hassan Masroor - EPCL Plant                       NaN                NaN  2023-10-02 00:00:00  2023-10-02 00:00:00            NaT  Self Learnings / Conclusio...  Add if any, Self Learnings...                            NaN  Good PPEs Compliance  Overall compliance was goo...                 HPO                 NaN                NaN                      NaN                      NaN                            NaN                               NaN                            NaN                      NaN                   NaN                 NaN                         NaN                            0

================================================================================

Sheet: Relationships
Number of columns: 5
Columns:
   1. source_type
   2. source_id
   3. target_type
   4. target_id
   5. relationship_type

First 5 rows of data:
  source_type        source_id        target_type            target_id relationship_type
0    Incident  IN-20220405-001  Corrective_Action  AC-INC-20231005-019         generates
1    Incident  IN-20220405-001  Corrective_Action  AC-INC-20231005-007         generates
2    Incident  IN-20220405-001  Corrective_Action  AC-INC-20231005-020         generates
3    Incident  IN-20220405-001  Corrective_Action  AC-INC-20231005-022         generates
4    Incident  IN-20220405-001  Corrective_Action  AC-INC-20231005-023         generates

================================================================================

Sheet: Location_Summary
Number of columns: 3
Columns:
   1. location
   2. sheets_involved
   3. sheet_list

First 5 rows of data:
                      location  sheets_involved                     sheet_list
0                           CA                4  Audit, Audit Findings, Ins...
1       Manufacturing Facility                4  Audit, Audit Findings, Ins...
2              Asset Integrity                4  Audit, Audit Findings, Ins...
3  Engro Polymer and Chemicals                4  Audit, Audit Findings, Ins...
4                  Power Plant                4  Audit, Audit Findings, Ins...

================================================================================

Sheet: Department_Summary
Number of columns: 3
Columns:
   1. department
   2. sheets_involved
   3. sheet_list

First 5 rows of data:
                      department  sheets_involved           sheet_list
0            Process - EDC / VCM                2  Incident, Hazard ID
1                   Not Assigned                2  Incident, Hazard ID
2                            PVC                2  Incident, Hazard ID
3  Chlor Alkali and Allied Ch...                1             Incident
4                  Process - PVC                2  Incident, Hazard ID

================================================================================

