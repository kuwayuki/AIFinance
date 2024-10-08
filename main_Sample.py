import utils
import sys
from prompts import PROMPT_SYSTEM, PROMPT_USER

def main():
    ticker = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    prompt = PROMPT_USER["LEADER"].format(
        ticker=ticker,
    )
    response = utils.get_ai_opinion(prompt, PROMPT_SYSTEM["IS_TRUE"])
    # response = utils.get_industry_tickers(ticker)
    print(response)

if __name__ == "__main__":
    main()

