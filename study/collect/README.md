# Collecting answers automatically

**Set up on 2026-11-21:** sheet "ego-vision responses" in Harshil's personal
Google account; the web app address is in `study/endpoint.js`.

Without this, raters copy a code and labellers download a file, and both
send it to you by hand. With it, the pages send the answers to a Google
Sheet that only you can open. The hand route stays as a backup.

## Setup (once, about 5 minutes)

Use a personal Google account. A college Google Workspace account often
blocks web apps that "Anyone" can reach.

1. Open <https://sheets.new>. Name the sheet `ego-vision responses`.
2. In the sheet: **Extensions > Apps Script**. Delete the code that is
   there, paste the whole of `study/collect/apps_script.gs`, press Save.
3. **Deploy > New deployment**. Click the gear next to "Select type" and
   choose **Web app**.
   - Description: `collect`
   - Execute as: **Me**
   - Who has access: **Anyone**
4. Press **Deploy**. Google asks you to authorise the script: choose your
   account, then "Advanced", then "Go to ... (unsafe)", then "Allow". (It
   says "unsafe" only because you wrote the script yourself and Google has
   not reviewed it. The script can only add rows to this one sheet.)
5. Copy the **Web app URL** (it ends in `/exec`).
6. Put it in `study/endpoint.js`:
   `window.STUDY_ENDPOINT = "https://script.google.com/macros/s/.../exec";`
   then commit and push. Both the rating study and all labelling pages
   use it at once; nothing has to be rebuilt.
7. Test: open the URL in a browser. It should show
   `{"ok":true,"note":"collection endpoint is running"}`.

The sheet gets two tabs, `ratings` and `labels`, one row per person.
Apps Script does not give the script the sender's IP address or Google
account, so the rating rows stay anonymous. Only the date is stored.

## Getting the answers out

**File > Download > Comma-separated values**, once per tab.

- `ratings` tab -> `study/responses_v2/ratings.csv` (anonymous; it may be
  committed, as the pilot's answers were)
- `labels` tab -> `runs/label_check/relabel/answers/labels.csv` (holds
  the labellers' names; never commit it)

Codes and files that people send by hand go to the same folders
(`.txt` with one code per line, or `.json`). The analysis scripts read
all of them and keep each person once.
