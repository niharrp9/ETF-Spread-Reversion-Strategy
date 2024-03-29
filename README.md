# ETF Spread-Reversion Strategy

This repository houses the Python code for a spread-reversion trading strategy tailored to ETF pairs. The strategy identifies and capitalizes on the spread in M-day returns of the ETF Pairs, going long on the undervalued ETF and shorting the overvalued counterpart when the spread is to be bought and vice versa. Position sizes are dynamically adjusted daily based on the median dollar volume to ensure capital efficiency.

## Features
- **Data Analysis**: Involves processing historical split- and dividend-adjusted closing prices and daily Fama-French factor returns from December 2, 2021, to November 15, 2023.
- **Algorithmic Trading**: Implements entry and exit logic based on return disparities, with a stop-loss mechanism to preserve capital.
- **Performance Tracking**: Monitors real-time profits and losses for individual positions and cumulatively across the portfolio.
- **Risk Management**: A stop-loss protocol triggers an exit if losses exceed a predetermined percentage of the trade value, with a trading pause for the rest of the month.
- **Trading Cost Management**: Simulates real-world trading by factoring in variable trading costs, analyzing their impact on the overall strategy performance.
- **Parameter Tuning**: Engages in parameter optimization to enhance strategy outcomes, allowing for variations in market scenarios, including zero-cost trades.
- **Quantitative Analysis**: Leverages statistical methods to assess the strategy's alignment with market factors like Fama-French factor returns and volatility levels.
- **Visualization**: Utilizes data visualization tools to graphically represent the strategy's performance, offering insights into its behavior under various parameter configurations.

The repository provides a comprehensive overview of the strategy's mechanics and its performance implications, offering a resource for those interested in quantitative finance and trading algorithms.

## Usage
Feel free to explore the strategy, test it with your own data, or contribute to its improvement. For a detailed explanation of the strategy and its outcomes, please refer to the [Project HTML] section.

## Contributions
Contributions, issues, and feature requests are welcome. To contribute, simply fork the repository, create your feature branch, commit your changes, and open a pull request.

## Contact
For any questions or discussions regarding the strategy, reach out through the [Discussions] tab or contact me directly via my [email].

[Project HTML]: # (Link to the visualization section in your repository)
[Discussions]: # (Link to the discussions page in your repository)
[email]: # (mailto:pandanihar1996@gmail.com)
