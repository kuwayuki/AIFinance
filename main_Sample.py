import utils
import sys
from prompts import PROMPT_SYSTEM, PROMPT_USER

ticker = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
tickers = [ticker.strip().upper() for ticker in sys.argv[1].split(',')] if len(sys.argv) > 1 else ['9697.T']
prompt = PROMPT_USER["LEADER"].format(
    ticker=ticker,
)
response = utils.sample(tickers)
# response = utils.get_ai_opinion(prompt, PROMPT_SYSTEM["IS_TRUE"])
# response = utils.get_industry_tickers(ticker)
print(response)

