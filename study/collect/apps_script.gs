/**
 * Collection web app for the rating study (study/rate/) and the labelling pages (study/label/).
 *
 * Each POST body is one JSON result. It is appended as one row to the tab named after its
 * "kind" ("ratings" or "labels"). Apps Script does not pass the sender's IP address or
 * Google account to the script, so the rating rows stay anonymous. Only the date is
 * stored, not the time of day.
 *
 * Setup (once, about 5 minutes): see study/collect/README.md.
 */
const KINDS = ["ratings", "labels"];
const MAX = 45000;  // a Sheets cell holds at most 50,000 characters

function doPost(e) {
  const lock = LockService.getScriptLock();
  lock.waitLock(20000);
  try {
    const body = (e && e.postData && e.postData.contents) || "";
    let d;
    try { d = JSON.parse(body); } catch (err) { return reply({ ok: false, error: "not JSON" }); }
    const kind = KINDS.indexOf(d.kind) >= 0 ? d.kind : "other";
    if (body.length > MAX) return reply({ ok: false, error: "too long" });
    const ss = SpreadsheetApp.getActiveSpreadsheet();
    const sh = ss.getSheetByName(kind) || ss.insertSheet(kind);
    if (sh.getLastRow() === 0) sh.appendRow(["date", "kind", "version", "id", "json"]);
    const day = Utilities.formatDate(new Date(), "Asia/Kolkata", "yyyy-MM-dd");
    sh.appendRow([day, kind, String(d.v || ""), String(d.id || d.run || ""), body]);
    return reply({ ok: true });
  } finally {
    lock.releaseLock();
  }
}

function doGet() {
  return reply({ ok: true, note: "collection endpoint is running" });
}

function reply(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}
