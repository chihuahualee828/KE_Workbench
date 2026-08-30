# Demo Script

1. Start the app with `docker compose up --build`.
2. Open `http://localhost:3000`.
3. Confirm the API status reads `API connected`.
4. In the workbench, run generation for `The Space Needle is located in the city of`.
5. Open `Logit Lens`, keep target token ` Seattle`, and run the lens.
6. Open `Trace`, use subject `The Space Needle` and expected answer `Seattle`, then run tracing.
7. Open `Edit`, apply the ROME form with new object `Paris`.
8. Switch to `Patches` and note the saved patch ID.
9. Open `Generate`, select that patch, and run the same prompt.
10. Open `Evaluate`, select the patch, and run evaluation.

ROME/MEMIT require EasyEdit; use `INSTALL_EASYEDIT=true docker compose up --build` or set `EASYEDIT_ROOT` to a local EasyEdit checkout before applying edits.
