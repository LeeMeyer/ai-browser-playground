# Local AI + LiveCodes Playground

Small browser demo:

- LiveCodes on the left
- Chat agent on the right
- Two providers: Chrome built-in AI or Transformers.js

The agent can read and update HTML/CSS/JS in the playground.

## Run

```powershell
npx --yes serve -l 5173
```

Open http://localhost:5173 in Chrome.

## Use

1. Pick a provider.
2. If using Transformers.js, pick a model from the dropdown.
3. Click Initialize AI.
4. Send a prompt and the coding playground will be updated


## Notes

- First model load can take a while.
- Chrome built-in AI depends on your Chrome/device support.
- have to meet the system requirements for the models to be able to run
