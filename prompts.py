PROMPT_SYSTEM = {
    "NONE": "",
    "ARRAY": "配列で返却してください",
    "IS_TRUE": "True or Falseで返却してください",
    "JAPANESE_SUMMARY_ARRAY": "日本語で要約して配列で返却してください。先頭に日付も記載してください。",
    "JAPANESE_SUMMARY": "日本語で要約して配列で返却してください",
}

PROMPT_USER = {
    "TOP5": "{ticker}と同じ業界・セクターと同じ銘柄のシンボルのみを上から順に売上トップ5（{ticker}を含む）で教えてください",
    "LEADER": "{ticker}と同じ業界またはセクターの中で、{ticker}がCANSLIMのLになりうる独自の製品があるかを教えてください",
    "NEW_PRODUCT": """下記の中で馴染みのないワードがあるかを教えてください
    {product}
    """,
    "SAMPLE": """
    """,
    "ANALYST_EVAL": """
    下記は{ticker}のアナリストの評価データです。
    客観的に下記データを評価して、1~7の将来的な株価についての情報を教えてください。
    ただし、Analyst Price Targetsはトレンドによっては参考になりませんのでその時は除外してください

    {analyst}

    1~3は10点満点でお願いします。
    1. 評価数
    2. 短期的な株価の上昇性
    3. 長期的な株価の上昇性
    4. 理想的な売り価格（上昇率%）と到達する確率%、時期
    5. 現実的な売り価格（上昇率%）と到達する確率%、時期
    6. 推奨する購入価格
    7. 損切価格

    今日は{current_date}です。
    """,
}

PROMPT_BASE_ALL = """
私は{ticker}の株を保有しています。下記の情報から、株価に影響を与える可能性が高いニュースを日時も考慮して複数選び、その理由と影響について簡潔に教えてください。
今日は{current_date}です。最後に、全体の結論も、特に重要な要因に基づいて論理的に導いてください。

最近のニュースヘッドライン:
{news_headlines}
{historical_section}

また、以下の銘柄について、短期的および長期的に推奨される株を、それぞれ具体的なデータに基づいて教えてください。
その理由も簡潔に述べてください:
{osusume_stocks}
"""

PROMPT_BASE_DETAIL = """
私は{ticker}の株を保有しています。下記の情報からこの銘柄の将来の動向に関する分析を簡潔に提供してください。
今日は{current_date}です。

下記の情報から、この銘柄に影響を与える可能性が高いニュースを日時も考慮して複数注目してください。
また、過去のデータを基にしたトレンドやパターンを特定し、それに基づく今後の予測を行ってください。
可能であれば、類似の市場状況があった際の過去のパフォーマンスも考慮してください。

さらに、以下の点を考慮して分析してください：
- **ファンダメンタル分析**: 企業の財務状況、業績、利益率、PER、株主への配当などの基礎的な指標に基づいて、この銘柄の中長期的な成長性やリスクを評価してください。
- **テクニカル分析**: 株価のチャートや移動平均線、出来高、ボリンジャーバンド、MACDなどの指標を用いて、価格変動のパターンやトレンドを特定し、今後の動向を予測してください。

最近のニュースヘッドライン:
{news_headlines}
{historical_section}

特に以下の点について具体的な数値を用いて簡潔に説明してください。：
1. 将来のパフォーマンスに影響を与える可能性のある主要なニュースとその理由。
2. 直近のデータを特に重視した将来の予測。上昇、下降する場合は、どれくらいの期間が見込まれるのか。どのニュースを元に回答したか。
3. 買い時と売り時はいつか。具体的に何月ごろか。損切価格も設定するとしたらいくらか。
"""

PROMPT_BASE_SHORT = """
2週間以内の短期投資で買いが推奨される有名で安全な米国ETFと個別の米国銘柄を、下記のニュースから日時も考慮して複数教えてください。ニュースの日時も返答してください。
今日は{current_date}です。ティッカーシンボル部分のみ「」で囲ってください。

最近のニュースヘッドライン:
{news_headlines}
{historical_section}
"""

PROMPT_RELATIONS_CUT = """
下記のニュースで米国株価予測に関係しそうな行のみを全て返却してください。関係がない説明や重複する内容、引用元会社名は不要です。
"""
# 下記のニュースで米国株価予測に関係しそうな行のみを改行区切りで全て返却してください。余計な説明や引用元会社名は不要です。

PROMPT_BASE_PROMISING = """
私は{ticker}の株購入を検討しています。
四季報を参照した選定基準は下記で、5年以内に50%以上の利益を得たいと考えていますが、こちらに対しての冷静な意見をお願いします。

1.3期平均の成長率が10%以上
2.売上高／利益率が10%以上
3.予想PERが20%以下
4.ROEが17%以上

最近のニュースヘッドライン（信頼性の高いソースからのものも含む）:
{news_headlines}
{historical_section}

また、{ticker}の成長性について、ホームページの内容から将来的にどのように評価されますか？
{company_homepage_url}

今日は{current_date}です。上記の内容を考慮し、{ticker}の購入に関してご意見をお聞かせください。
"""

PROMPT_SYSTEM_BASE = """
あなたは株式市場の予測に精通した金融アナリストです。提供された情報を基に、論理的かつ簡潔に分析を行ってください。具体的なデータや数値に基づいて客観的な結論を導き、主観的な表現を避けてください。分析には以下を含めてください:

- マクロ経済指標や業界トレンド
- 競合他社との比較とベンチマーク
- 詳細な財務比率の分析（キャッシュフロー、負債比率、フリーキャッシュフローなど）
- 潜在的なリスク要因や不確実性の定量化
- 成長ドライバーや新規事業計画
- 経営陣の質や技術革新能力
- 技術的指標の分析（チャートパターン、移動平均線、RSIなど）
- 最新のニュースや最近の発表
- 企業の倫理的・法的観点、ESG要素
- シナリオ分析とストレステスト

また、**購入・売却のタイミングや価格に関する情報**も提供してください。
"""


PROMPT_PROMISING_FUTURE = """
私は{ticker}の株購入を検討しています。この株を購入して5年以内に大きな利益を得たいと考えていますが、妥当でしょうか？
過去のニュースや最新の情報から{ticker}の将来性を導き出し、この企業は成長できるかを冷静に分析してください。
今日は{current_date}です。

・四季報の選定基準
1. 3期の成長率が10%以上
2. 売上高／利益率が10%以上
3. 予想PERが20%以下
4. ROEが17%以上

・企業情報
{research}

{historical_section}

・過去のニュース（時系列も重視してください）
{news}

・追加で考慮してほしい事項
- マクロ経済指標や業界トレンドを考慮してください。
- 主要な競合他社との比較を行い、強みと弱みを評価してください。
- キャッシュフローや負債比率などの財務指標も含めてください。
- 潜在的なリスク要因や不確実性を考慮してください。
- 今後の成長を促進する要因や新規事業計画を評価してください。
- 経営陣の質や技術革新能力などの定性的な要素も評価してください。
- チャートパターンや取引量などの技術的指標も考慮してください。
- 最新のニュースや最近の発表も含めて分析してください。
- 企業の倫理的な取り組みや法的な問題も評価してください。

・教えてほしい情報
1. 5年以内に大きな利益がでる、将来性はあるかの簡潔な解説
2. 購入のタイミングと価格（アナリスト評価も少しだけ参考）
3. 損切のタイミングと価格
4. 最短で最大の利益が得られるパフォーマンスの良い売却タイミングと価格
5. 企業や商品の簡易な説明と今後の戦略
"""

PROMPT_SYSTEM_GROW = """
あなたは株式市場の予測に精通した金融アナリストです。提供された情報を基に、論理的かつ簡潔な分析を行ってください。具体的なデータや数値に基づき、マクロ経済指標、業界トレンド、競合分析、財務指標、リスク要因、成長ドライバー、定性的要素、技術的指標、最新ニュース、倫理的・法的観点を含めてください。回答には「*」や「#」などの記号や引用元名を使用せず、客観的な視点で分析を行ってください。
"""

PROMPT_PROMISING_GROW = """
私は有望な投資先を探しています。以下の企業情報と過去のニュースを基に、テンバガーとなりそうな有力な銘柄のみを教えてください。

・企業情報一覧
{research}

・過去のニュース（時系列も重視してください）
{news}

追加で考慮してほしい事項:
- マクロ経済指標や業界トレンドの分析
- 主要な競合他社との比較と相対的な強み・弱みの評価
- キャッシュフローや負債比率などの財務指標
- 潜在的なリスク要因や不確実性の考慮
- 今後の成長を促進する要因や新規事業計画の評価
- 経営陣の質や技術革新能力などの定性的要素
- 5年以内に大きな利益を得るための具体的なシナリオ
- チャートパターンや取引量などの技術的指標
- 最新のニュースや最近の発表の分析
- 企業の倫理的な取り組みや法的な問題の評価

有望な投資先の情報として以下を提供してください:
1. 5年以内に大きな利益が出る将来性のある銘柄の簡潔な解説
2. 購入のタイミングと具体的な価格（アナリスト評価も含む）
3. 損切のタイミングと具体的な価格
4. 最短で最大の利益が得られる売却タイミングと価格
5. GARP投資の評価
6. ピオトロスキー・スコアの評価
7. マジックフォーミュラ投資の評価
"""

PROMPT_CSV_TO_COMPACT_SYSTEM = """
各行を指示された内容に修正し、文字列の配列のみを返却してください。
"""

PROMPT_CSV_TO_COMPACT = """
下記は株価に関係のあるニュースです。これらのニュースから未来予測をしたいので、未来予測ができる内容のみコンパクトにまとめてください。
"""

PROMPT_CAN_SLIM_SYSTEM = """
有望なTickerシンボルのみを文字列の配列のみで返却してください。なければ解説をお願いします。
"""
# 有望なTickerシンボルのみを文字列の配列のみで返却してください。

PROMPT_CAN_SLIM_USER = """
下記はCAN-SLIM法で有望な銘柄一覧の候補です。以下の企業情報と過去のニュースを基に、CAN-SLIM法に該当する銘柄のみを求めたいです。
{research}

・過去のニュース
{news}

今日は{current_date}です。
"""