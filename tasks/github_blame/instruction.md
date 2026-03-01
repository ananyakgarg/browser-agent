For each repo and code_string in the CSV:

1. Find the file in the repo that contains the code_string.

2. Navigate to the BLAME VIEW for that file. Find the line(s) matching the code_string.

3. Check how long ago the last change to that code was.

4. If the code was modified within the past year:
   - Go into the commit and read the diff.
   - Determine if it may have materially changed how something was calculated.
   - If so, flag it and take screenshots to document.

5. If the code was NOT modified in the past year:
   - Check the code BELOW the function — look for code that changes how the function operates that WAS modified within the past year.
   - If found, determine if this could have materially affected the outcome of the function/calculation.
   - Document with screenshots if so.

Output fields: file_path, line_number, commit_hash, commit_author, commit_date, blame_url, modified_within_year, material_change_flag, material_change_explanation, evidence_screenshots
