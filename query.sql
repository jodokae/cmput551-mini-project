SELECT DISTINCT
<<<<<<< HEAD
gh_num_commits_in_push,
=======
gh_lang,
gh_num_commits_in_push,
git_prev_commit_resolution_status,
>>>>>>> a7d033adf2fc995315d30f4a8c5c3c751fb4a742
gh_team_size,
git_num_all_built_commits,
gh_num_issue_comments,
gh_num_commit_comments,
gh_num_pr_comments,
git_diff_src_churn,
git_diff_test_churn,
gh_diff_files_added,
gh_diff_files_deleted,
gh_diff_files_modified,
gh_diff_tests_added,
gh_diff_tests_deleted,
gh_diff_src_files,
gh_diff_doc_files,
gh_diff_other_files,
gh_num_commits_on_files_touched,
gh_sloc,
gh_test_lines_per_kloc,
gh_test_cases_per_kloc,
gh_asserts_cases_per_kloc,
gh_by_core_team_member,
gh_description_complexity,
<<<<<<< HEAD
=======
gh_pushed_at,
gh_build_started_at,
>>>>>>> a7d033adf2fc995315d30f4a8c5c3c751fb4a742
tr_duration,
tr_log_bool_tests_ran,
tr_log_bool_tests_failed,
tr_log_num_tests_ok,
tr_log_num_tests_failed,
tr_log_num_tests_run,
tr_log_num_tests_skipped,
tr_log_testduration,
tr_log_buildduration,
<<<<<<< HEAD
gh_lang,
git_prev_commit_resolution_status,
gh_pushed_at,
gh_build_started_at,
tr_status
FROM travistorrent_8_2_2017
WHERE tr_status = "errored" OR tr_status = "failed" OR tr_status = "passed";
=======
tr_status
FROM travistorrent_8_2_2017
WHERE tr_status = "errored" OR tr_status = "failed" OR tr_status = "passed"
LIMIT 100000;
>>>>>>> a7d033adf2fc995315d30f4a8c5c3c751fb4a742
