// Tests for the Node.js frontend server.   Run with:  npm test
import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { createServer } from "../server.js";

function listen(server) {
  return new Promise((resolve) => {
    const s = server.listen(0, "127.0.0.1", () => resolve(s));
  });
}
const urlOf = (s) => `http://127.0.0.1:${s.address().port}`;

test("serves the chat page", async () => {
  const s = await listen(createServer({ backendUrl: "http://127.0.0.1:9" }));
  const res = await fetch(urlOf(s) + "/");
  const html = await res.text();
  assert.equal(res.status, 200);
  assert.match(html, /<title>NetraChat<\/title>/);
  s.close();
});

test("forwards /api requests to the Python backend", async () => {
  // A stand-in for api.py that echoes what it received.
  const fake = await listen(http.createServer((req, res) => {
    let body = "";
    req.on("data", (c) => { body += c; });
    req.on("end", () => {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ path: req.url, method: req.method, body }));
    });
  }));
  const s = await listen(createServer({ backendUrl: urlOf(fake) }));
  const res = await fetch(urlOf(s) + "/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: "hi" }),
  });
  const data = await res.json();
  assert.equal(data.path, "/api/chat");
  assert.equal(data.method, "POST");
  assert.deepEqual(JSON.parse(data.body), { message: "hi" });
  s.close();
  fake.close();
});

test("explains clearly when the backend is not running", async () => {
  const s = await listen(createServer({ backendUrl: "http://127.0.0.1:9" }));
  const res = await fetch(urlOf(s) + "/api/health");
  assert.equal(res.status, 502);
  assert.match((await res.json()).detail, /python api\.py/);
  s.close();
});
