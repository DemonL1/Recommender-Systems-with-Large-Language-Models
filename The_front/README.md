Movie Recommender (Frontend)
============================

Overview
--------
This is a small static frontend for requesting movie recommendations from a configured backend API. The page allows a user to enter movies or preferences, sends a prompt-enhanced request to the backend, and displays recommendations along with an in-page conversation history.

Key Features
------------
- Modern, responsive single-page UI (card layout).
- Adds a prompt prefix to each request to ensure the backend returns English-only recommendations without chain-of-thought.
- Keeps an in-memory conversation history (browser memory) showing timestamp, user input, and recommendation.

Files
-----
- `test.html` — Main static page. Edit this file to change UI, the prompt prefix, or the target API.

Configuration & Usage
---------------------
1. Open `test.html` in a web browser (double-click or serve via a simple static server).
2. Edit the `API_URL` constant inside `test.html` to point to your recommendation backend endpoint.
3. Enter one or more movies or preferences in the input box and click "Get Recommendation".

Prompt Behavior
---------------
The client automatically prepends a fixed prompt (see `PROMPT_PREFIX` in `test.html`) to the user's input before sending the request. This ensures consistent instructions to the backend (for example: "Provide recommendations and reasons in English, do not show chain-of-thought.").

Notes about Conversation History
-------------------------------
- The history is stored only in the browser memory for the current page session. Reloading the page will clear the history.
- If you want persistent history, you can modify `test.html` to send/save entries to a server or to `localStorage`.

Extending the Project
---------------------
- To persist history: update the client to save `history` into `localStorage` or POST entries to a backend endpoint.
- To change language or prompt rules: modify `PROMPT_PREFIX` in `test.html`.
- To format or restrict results: post-process the server response in `test.html` before rendering.

License
-------
This project is provided as-is for demonstration purposes.


