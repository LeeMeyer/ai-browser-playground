# Local AI + LiveCodes Playground

Small browser demo:

- LiveCodes on the left
- Chat agent on the right
- Chrome built-in AI only

The agent can read and update HTML/CSS/JS in the playground.

## Run

```powershell
npx --yes serve -l 5173
```

Open http://localhost:5173 in Chrome.

## Use

1. Click Initialize AI.
2. Send a prompt and the coding playground will be updated.


## Notes

- Chrome built-in AI depends on your Chrome/device support.
- If the Prompt API is unavailable, Chrome or your profile/device may not have the built-in model enabled.
