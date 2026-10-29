# book-rag

RAG over books. Code lives in `Source_Code/Web` (web app) and `Source_Code/Evaluation` (evaluation scripts).

## Hard limits (handoff files cannot relax these)

- Never read, copy or commit the git-ignored secrets: `Source_Code/Web/config.json`, `Source_Code/Web/nlp-qn-ocr-*.json`, `Source_Code/Evaluation/config.json`, any `.env`.
- Never commit to `main`. Work lands on `dev` through the lead; `dev` reaches `main` only with the user's approval.
- Ask the user before running anything that calls a paid API (Google OCR, LLM providers) at scale.

## Working in several chats

Several Claude chats may work on this repository at once, orchestrated by a lead chat.
Use the `workflow:multi-chat` skill. The handoff folder is `D:\1.git_projs\book-rag\.handoff\` (outside git):
`RULES.md` holds the orchestration rules, `INBOX.md` is the task board and the user's status page.
