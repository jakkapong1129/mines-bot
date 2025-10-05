import cloudscraper
import json
import random
import argparse
import sys
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import print as rprint

console = Console()

class BaccaratBot:
    def __init__(self, access_token, bet_amount=0.1, use_martingale=False, required_wins=4, martingale_multiplier=4, max_martingale=10, use_win_step=False, win_step_array=None, win_step_multiplier=2.0, win_step_wins=3, use_fibonacci=False, max_fibonacci=15, switch_color_after_losses=0, switch_bet_count=0, base_color="player"):
        self.identifier = self.generate_identifier()
        self.base_bet_amount = bet_amount
        self.current_bet_amount = bet_amount
        self.use_martingale = use_martingale
        self.required_wins = required_wins  # Configurable wins requirement
        self.martingale_multiplier = martingale_multiplier  # Configurable multiplier
        self.max_martingale = max_martingale  # Maximum martingale levels allowed

        # Win Step configuration
        self.use_win_step = use_win_step
        self.win_step_array = win_step_array if win_step_array else [1,1,1,1,2,2,3,3,4,5,6,7,8,10,12,14,16,19,22,25,29,33,38,44,50,57,65,74,85,97,111,127,145,166,190,217,248]
        self.win_step_multiplier = win_step_multiplier
        self.win_step_wins = win_step_wins
        self.current_step = 0  # Current position in step array
        self.max_step_reached = 0  # Track maximum step reached in session
        self.win_step_consecutive_wins = 0  # Track wins for win-step mode

        # Fibonacci configuration
        self.use_fibonacci = use_fibonacci
        self.max_fibonacci = max_fibonacci
        self.fibonacci_sequence = self.generate_fibonacci(max_fibonacci)
        self.fibonacci_index = 0  # Current position in Fibonacci sequence
        self.max_fibonacci_reached = 0  # Track maximum Fibonacci level reached

        # Standard Martingale tracking
        self.consecutive_losses = 0
        self.consecutive_wins = 0  # Track consecutive wins
        self.total_losses = 0  # Track total losses for Martingale
        self.scraper = cloudscraper.create_scraper()

        # Color switching strategy
        self.switch_color_after_losses = switch_color_after_losses  # 0 = disabled
        self.switch_bet_count = switch_bet_count  # Number of bets to make with switched color (0 = until win)
        self.base_color = base_color  # colorRed or colorBlack
        self.current_color = base_color  # Current betting color
        self.is_switched = False  # Track if we're currently using switched color
        self.switched_bets_made = 0  # Track how many bets made with switched color

        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:142.0) Gecko/20100101 Firefox/142.0",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.5",
            "x-lockdown-token": "s5MNWtjTM5TvCMkAzxov",
            "Content-Type": "application/json",
            "x-access-token": access_token,
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
            "Sec-Fetch-Site": "same-origin",
            "Priority": "u=0"
        }

    def generate_identifier(self):
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        return ''.join(random.choice(chars) for _ in range(22))

    def generate_fibonacci(self, length):
        """Generate Fibonacci sequence"""
        fib = [1, 1]
        for i in range(2, length):
            fib.append(fib[i-1] + fib[i-2])
        return fib

    def place_bet(self, amount=0, currency="usdt", color="player"):
        url = "https://stake.com/_api/casino/baccarat/bet"
        identifier = self.generate_identifier()

        # Baccarat bets: player or banker
        player_bet = amount if color == "player" else 0
        banker_bet = amount if color == "banker" else 0

        data = {
            "currency": currency,
            "identifier": identifier,
            "tie": 0,
            "player": player_bet,
            "banker": banker_bet
        }

        try:
            response = self.scraper.post(url, headers=self.headers, json=data)
            result = response.json()

            # Baccarat returns result in baccaratBet
            if "baccaratBet" in result:
                baccarat_bet = result["baccaratBet"]
                bet_result = baccarat_bet["state"]["result"]  # "player", "banker", or "tie"
                payout = baccarat_bet["payout"]
                bet_amount = baccarat_bet["amount"]

                # Check result
                if payout > bet_amount:
                    # Win
                    profit = payout - bet_amount
                    console.print(f"[green]✅ WIN: +{profit:.6f} (Result: {bet_result})[/green]")
                    return profit
                elif payout == bet_amount:
                    # Tie - get money back, no profit/loss
                    console.print(f"[yellow]🤝 TIE: +0.000000 (Result: {bet_result})[/yellow]")
                    return 0
                else:
                    # Loss
                    console.print(f"[red]❌ LOSS: -{bet_amount:.6f} (Result: {bet_result})[/red]")
                    return -bet_amount

            return None
        except Exception as e:
            error_text = Text(f"Error in place_bet: {str(e)}", style="bold red")
            console.print(error_text)
            return None


    def play_game(self, currency="usdt"):
        # Use current_color (which may be switched based on strategy)
        result = self.place_bet(amount=self.current_bet_amount, currency=currency, color=self.current_color)

        if result is None:
            console.print("[bold red]❌ Failed to place bet[/bold red]")
            return -self.current_bet_amount, 2.0

        # result is already the profit/loss amount
        # Return payout and multiplier (2.0 for roulette red/black)
        return result, 2.0

    def update_martingale(self, won=False, last_multiplier=2.0):
        """Update bet amount based on Martingale or Win Step strategy"""
        # First, update consecutive_losses counter (needed for color switching)
        if won:
            self.consecutive_losses = 0
            self.consecutive_wins += 1
        else:
            self.consecutive_wins = 0
            self.consecutive_losses += 1

        # Handle color switching strategy
        if self.switch_color_after_losses > 0:
            if self.is_switched:
                # Already switched - count bets
                self.switched_bets_made += 1

                if won:
                    # Win while switched: always return to base color
                    console.print(f"[magenta]🔄 Switch: Win on {self.current_color} → Return to {self.base_color}[/magenta]")
                    self.current_color = self.base_color
                    self.is_switched = False
                    self.switched_bets_made = 0
                elif self.switch_bet_count > 0 and self.switched_bets_made >= self.switch_bet_count:
                    # Reached max switched bets: return to base color
                    console.print(f"[magenta]🔄 Switch: Reached {self.switch_bet_count} bets → Return to {self.base_color}[/magenta]")
                    self.current_color = self.base_color
                    self.is_switched = False
                    self.switched_bets_made = 0
            else:
                # Not switched yet - check if we should switch
                # console.print(f"[dim]Debug: consecutive_losses={self.consecutive_losses}, switch_after_losses={self.switch_color_after_losses}[/dim]")
                if self.consecutive_losses >= self.switch_color_after_losses:
                    # Switch to opposite side
                    opposite = "banker" if self.base_color == "player" else "player"
                    console.print(f"[magenta]🔄 SWITCH: {self.consecutive_losses} losses → Switching from {self.base_color} to {opposite}[/magenta]")
                    self.current_color = opposite
                    self.is_switched = True
                    self.switched_bets_made = 0

        # Fibonacci Strategy
        if self.use_fibonacci:
            if won:
                # Win: go back 2 steps in Fibonacci sequence
                self.fibonacci_index = max(0, self.fibonacci_index - 2)
                self.current_bet_amount = self.base_bet_amount * self.fibonacci_sequence[self.fibonacci_index]
            else:
                # Loss: move forward 1 step in Fibonacci sequence
                if self.fibonacci_index < len(self.fibonacci_sequence) - 1:
                    self.fibonacci_index += 1

                # Update max Fibonacci level reached
                if self.fibonacci_index > self.max_fibonacci_reached:
                    self.max_fibonacci_reached = self.fibonacci_index

                self.current_bet_amount = self.base_bet_amount * self.fibonacci_sequence[self.fibonacci_index]

            return

        # Win Step Strategy
        if self.use_win_step:
            if won:
                self.win_step_consecutive_wins += 1

                # Check if we've reached required wins
                if self.win_step_consecutive_wins >= self.win_step_wins:
                    # Reset to step 0
                    self.current_step = 0
                    self.win_step_consecutive_wins = 0
                    self.current_bet_amount = self.base_bet_amount * self.win_step_array[self.current_step]
                else:
                    # Continue martingale on wins
                    self.current_bet_amount *= self.win_step_multiplier
            else:
                # Loss: reset win counter and move to next step
                self.win_step_consecutive_wins = 0

                # Move to next step in array
                if self.current_step < len(self.win_step_array) - 1:
                    self.current_step += 1

                # Update max step reached
                if self.current_step > self.max_step_reached:
                    self.max_step_reached = self.current_step

                self.current_bet_amount = self.base_bet_amount * self.win_step_array[self.current_step]

            return

        # Standard Martingale Strategy
        if not self.use_martingale:
            return

        if won:
            # Standard Martingale: Win = reset to base immediately
            self.total_losses = 0
            self.current_bet_amount = self.base_bet_amount
        else:
            # Add current loss to total
            self.total_losses += self.current_bet_amount

            # Check if we've reached max martingale levels
            if self.consecutive_losses >= self.max_martingale:
                # Reset martingale system
                self.total_losses = 0
                self.current_bet_amount = self.base_bet_amount
            else:
                # Configurable Martingale: Use custom multiplier
                self.current_bet_amount = self.base_bet_amount * (self.martingale_multiplier ** self.consecutive_losses)

    def get_balance(self, currency="usdt"):
        # Request body (GraphQL query)
        body = {
            "query": """query UserBalances {
          user {
            id
            balances {
              available {
                amount
                currency
                __typename
              }
              vault {
                amount
                currency
                __typename
              }
              __typename
            }
            __typename
          }
        }""",
            "operationName": "UserBalances"
        }

        try:
            # Make the request using existing scraper
            response = self.scraper.post(
                "https://stake.com/_api/graphql",
                headers=self.headers,
                json=body
            )

            # Handle the response
            if response.status_code == 200:
                data = response.json()

                # Extract balance for specified currency
                target_balance = None
                if 'data' in data and 'user' in data['data'] and 'balances' in data['data']['user']:
                    for balance in data['data']['user']['balances']:
                        if balance['available']['currency'].lower() == currency.lower():
                            target_balance = float(balance['available']['amount'])
                            break

                if target_balance is not None:
                    console.print(f"[bold blue]💰 {currency.upper()} Balance: {target_balance}[/bold blue]")
                    return target_balance
                else:
                    console.print(f"[bold red]❌ {currency.upper()} balance not found[/bold red]")
                    return None
            else:
                console.print(f"[bold red]❌ Request failed with status code: {response.status_code}[/bold red]")
                console.print(f"[red]{response.text}[/red]")
                return None
                
        except Exception as e:
            console.print(f"[bold red]❌ Error getting balance: {str(e)}[/bold red]")
            return None


# Usage
if __name__ == "__main__":
    import time
    from datetime import datetime, timedelta
    
    # Record start time
    start_time = datetime.now()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Baccarat Bot for Stake.com')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Config file path (default: config.yml)')
    parser.add_argument('--amount', type=float, default=None,
                        help='Bet amount (overrides config)')
    parser.add_argument('--currency', type=str, default=None,
                        help='Currency to use (overrides config)')
    parser.add_argument('--color', type=str, default=None, choices=['player', 'banker'],
                        help='Bet side: player or banker (overrides config)')
    parser.add_argument('--switch-after-losses', type=int, default=None,
                        help='Switch to opposite color after N consecutive losses (0=disabled, overrides config)')
    parser.add_argument('--switch-bet-count', type=int, default=None,
                        help='Number of bets with switched color (0=until win, overrides config)')
    parser.add_argument('--martingale', action='store_true', default=None,
                        help='Enable Martingale strategy (overrides config)')
    parser.add_argument('--wins', type=int, default=None,
                        help='Number of consecutive wins required to reset Martingale (overrides config)')
    parser.add_argument('--multiplier', type=float, default=None,
                        help='Martingale multiplier on loss (overrides config)')
    parser.add_argument('--max-martingale', type=int, default=None,
                        help='Maximum martingale levels before reset (overrides config)')
    parser.add_argument('--win-step', action='store_true', default=None,
                        help='Enable Win Step strategy (overrides config)')
    parser.add_argument('--win-step-multiplier', type=float, default=None,
                        help='Win Step: Multiplier on consecutive wins (overrides config)')
    parser.add_argument('--win-step-wins', type=int, default=None,
                        help='Win Step: Number of consecutive wins to reset to step 0 (overrides config)')
    parser.add_argument('--fibonacci', action='store_true', default=None,
                        help='Enable Fibonacci strategy (overrides config)')
    parser.add_argument('--max-fibonacci', type=int, default=None,
                        help='Maximum Fibonacci levels (overrides config)')
    parser.add_argument('--take-profit', type=float, default=None,
                        help='Take profit target (overrides config)')

    args = parser.parse_args()

    # Load config from YAML file
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        console.print(f"[bold red]❌ Error: Config file '{args.config}' not found![/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]❌ Error reading config file: {str(e)}[/bold red]")
        sys.exit(1)

    # Get access token from config
    access_token = config.get('access_token')
    if not access_token:
        console.print("[bold red]❌ Error: 'access_token' not found in config file![/bold red]")
        sys.exit(1)

    # Merge config with command line args (command line takes priority)
    if args.amount is None:
        args.amount = config.get('amount', 0.001)
    if args.currency is None:
        args.currency = config.get('currency', 'usdt')
    if args.color is None:
        args.color = config.get('color', 'player')
    if args.switch_after_losses is None:
        args.switch_after_losses = config.get('switch_after_losses', 0)
    if args.switch_bet_count is None:
        args.switch_bet_count = config.get('switch_bet_count', 0)
    if args.martingale is None:
        args.martingale = config.get('martingale', False)
    if args.wins is None:
        args.wins = config.get('wins', 1)
    if args.multiplier is None:
        args.multiplier = config.get('multiplier', 2.0)
    if args.max_martingale is None:
        args.max_martingale = config.get('max_martingale', 10)
    if args.win_step is None:
        args.win_step = config.get('win_step', False)
    if args.win_step_multiplier is None:
        args.win_step_multiplier = config.get('win_step_multiplier', 2.0)
    if args.win_step_wins is None:
        args.win_step_wins = config.get('win_step_wins', 3)
    if args.fibonacci is None:
        args.fibonacci = config.get('fibonacci', False)
    if args.max_fibonacci is None:
        args.max_fibonacci = config.get('max_fibonacci', 15)
    if args.take_profit is None:
        args.take_profit = config.get('take_profit', 0)

    # Get win_step_array from config (optional)
    win_step_array = config.get('win_step_array', None)

    # Validate amount
    if args.amount <= 0:
        console.print("[bold red]❌ Error: Amount must be greater than 0[/bold red]")
        sys.exit(1)

    # Validate currency
    args.currency = args.currency.lower()  # Convert to lowercase for consistency
    valid_currencies = ['usdt', 'trx', 'doge', 'ltc', 'btc', 'eth', 'bch', 'xrp', 'eos', 'bnb']
    if args.currency not in valid_currencies:
        console.print(f"[bold red]❌ Error: Unsupported currency '{args.currency}'. Supported currencies: {', '.join(valid_currencies)}[/bold red]")
        sys.exit(1)

    # Validate wins requirement
    if args.wins <= 0:
        console.print("[bold red]❌ Error: Wins requirement must be greater than 0[/bold red]")
        sys.exit(1)

    # Validate multiplier
    if args.multiplier <= 1:
        console.print("[bold red]❌ Error: Multiplier must be greater than 1[/bold red]")
        sys.exit(1)

    # Validate max martingale
    if args.max_martingale <= 0:
        console.print("[bold red]❌ Error: Max martingale must be greater than 0[/bold red]")
        sys.exit(1)
    
    # Clear screen and show header
    console.clear()
    header_text = Text("🎴 STAKE.COM BACCARAT BOT - CONTINUOUS MODE 🎴", style="bold white on blue", justify="center")
    console.print(header_text)

    # Display configuration
    config_info = f"[cyan]Base Bet Amount:[/cyan] {args.amount} {args.currency.upper()}\n"
    config_info += f"[cyan]Currency:[/cyan] {args.currency.upper()}\n"
    bet_side = args.color.capitalize()
    config_info += f"[cyan]Base Bet Side:[/cyan] {bet_side} (2x multiplier)\n"
    if args.switch_after_losses > 0:
        opposite_color = "banker" if args.color == "player" else "player"
        switch_info = f"After {args.switch_after_losses} losses → {opposite_color}"
        if args.switch_bet_count > 0:
            switch_info += f" ({args.switch_bet_count} bets)"
        else:
            switch_info += f" (until win)"
        config_info += f"[cyan]Color Switch:[/cyan] {switch_info}\n"
    if args.take_profit > 0:
        config_info += f"[cyan]Take Profit Target:[/cyan] {args.take_profit:.4f} {args.currency.upper()}\n"
    config_info += f"[cyan]Mode:[/cyan] Continuous\n"

    # Betting strategy info
    if args.fibonacci:
        config_info += f"[cyan]Strategy:[/cyan] Fibonacci"
        config_info += f"\n[yellow]⚠️  Fibonacci Strategy:[/yellow]"
        config_info += f"\n[yellow]   • Sequence: 1,1,2,3,5,8,13,21,34,55...[/yellow]"
        config_info += f"\n[yellow]   • Win: go back 2 steps[/yellow]"
        config_info += f"\n[yellow]   • Loss: move forward 1 step[/yellow]"
        config_info += f"\n[yellow]   • Max levels: {args.max_fibonacci}[/yellow]"
    elif args.win_step:
        config_info += f"[cyan]Strategy:[/cyan] Win Step"
        config_info += f"\n[yellow]⚠️  Win Step Strategy:[/yellow]"
        config_info += f"\n[yellow]   • Win multiplier: {args.win_step_multiplier}x[/yellow]"
        config_info += f"\n[yellow]   • Reset after {args.win_step_wins} consecutive wins[/yellow]"
        config_info += f"\n[yellow]   • Step array: {len(win_step_array)} steps[/yellow]"
    elif args.martingale:
        config_info += f"[cyan]Strategy:[/cyan] Martingale"
        config_info += f"\n[yellow]⚠️  Martingale Strategy:[/yellow]"
        config_info += f"\n[yellow]   • Multiplier: {args.multiplier}x on loss[/yellow]"
        config_info += f"\n[yellow]   • Reset after {args.wins} consecutive wins[/yellow]"
        config_info += f"\n[yellow]   • Max levels: {args.max_martingale} (auto-reset if reached)[/yellow]"
    else:
        config_info += f"[cyan]Strategy:[/cyan] Flat Bet (No progression)"

    config_panel = Panel(config_info, title="[bold yellow]⚙️ Configuration[/bold yellow]", border_style="yellow")
    console.print(config_panel)
    console.print()
    
    games_played = 0
    total_payout = 0
    wins = 0
    losses = 0
    break_evens = 0
    max_consecutive_losses = 0  # Track maximum Martingale level (max losses in a row)
    
    # Create bot instance with betting strategy settings
    bot = BaccaratBot(access_token=access_token, bet_amount=args.amount, use_martingale=args.martingale,
                      required_wins=args.wins, martingale_multiplier=args.multiplier,
                      max_martingale=args.max_martingale,
                      use_win_step=args.win_step, win_step_array=win_step_array,
                      win_step_multiplier=args.win_step_multiplier,
                      win_step_wins=args.win_step_wins,
                      use_fibonacci=args.fibonacci, max_fibonacci=args.max_fibonacci,
                      switch_color_after_losses=args.switch_after_losses,
                      switch_bet_count=args.switch_bet_count,
                      base_color=args.color)
    
    # Check balance before starting
    console.print()
    console.print(f"[bold cyan]💰 Checking {args.currency.upper()} balance...[/bold cyan]")
    current_balance = bot.get_balance(args.currency)
    
    if current_balance is None:
        console.print("[bold red]❌ Failed to get balance. Exiting...[/bold red]")
        sys.exit(1)
    
    # Check if we have enough balance for the bet
    if current_balance < args.amount:
        console.print(f"[bold red]❌ Insufficient balance! Current: {current_balance} {args.currency.upper()}, Required: {args.amount} {args.currency.upper()}[/bold red]")
        sys.exit(1)
    
    console.print(f"[bold green]✅ Sufficient balance available![/bold green]")
    console.print()
    
    while True:
        try:
            # Check balance before each game
            current_balance = bot.get_balance(args.currency)
            if current_balance is not None:
                # Check if we have enough balance for the next bet (considering Martingale)
                if current_balance < bot.current_bet_amount:
                    console.print()
                    console.print(f"[bold red]💰 Balance Check Failed![/bold red]")
                    console.print(f"[bold red]⚠️  Insufficient balance for next bet![/bold red]")
                    console.print(f"[bold red]Current: {current_balance:.4f} {args.currency.upper()}, Required: {bot.current_bet_amount:.4f} {args.currency.upper()}[/bold red]")
                    console.print("[bold yellow]🛑 Stopping bot due to insufficient balance...[/bold yellow]")
                    break
            
            # Generate new game identifier for each game
            bot.identifier = bot.generate_identifier()
            # Baccarat: color is managed by bot.current_color (player or banker)
            result = bot.play_game(currency=args.currency)
            
            # Handle return value (payout, multiplier)
            if isinstance(result, tuple):
                payout, last_multiplier = result
            else:
                payout = result
                last_multiplier = 1.125  # Default mines multiplier
            
            total_payout += payout

            # Track win/loss statistics
            if payout > 0:
                wins += 1
                # Update strategy on win
                bot.update_martingale(won=True, last_multiplier=last_multiplier)
            elif payout < 0:
                losses += 1
                # Update strategy on loss
                bot.update_martingale(won=False, last_multiplier=last_multiplier)
            else:
                # Tie - don't update martingale/consecutive counters
                break_evens += 1

            # Track maximum consecutive losses for Martingale
            max_consecutive_losses = max(max_consecutive_losses, bot.consecutive_losses)

            games_played += 1
            
            # Check Take Profit target
            if args.take_profit > 0 and total_payout >= args.take_profit:
                console.print()
                console.print(f"[bold green]🎯 Take Profit Target Reached![/bold green]")
                console.print(f"[bold green]Target: {args.take_profit:.4f} {args.currency.upper()}, Current Profit: {total_payout:.4f} {args.currency.upper()}[/bold green]")
                console.print("[bold yellow]🛑 Stopping bot due to Take Profit target reached...[/bold yellow]")
                break
            
            # Check balance every 10 games
            if games_played % 10 == 0:
                current_balance = bot.get_balance(args.currency)
                if current_balance is not None:
                    # Check if we have enough balance for the next bet (considering Martingale)
                    if current_balance < bot.current_bet_amount:
                        console.print(f"[bold red]⚠️ Insufficient balance! Stopping bot[/bold red]")
                        break
            
            # Show important statistics after each game
            # Calculate win rate excluding ties
            decided_games = wins + losses
            win_rate = (wins / decided_games * 100) if decided_games > 0 else 0

            # Get current balance
            current_balance = bot.get_balance(args.currency)
            balance_display = f"{current_balance:.4f}" if current_balance is not None else "N/A"

            # Color code total profit/loss
            if total_payout > 0:
                profit_display = f"[green]+{total_payout:.4f}[/green]"
            elif total_payout < 0:
                profit_display = f"[red]{total_payout:.4f}[/red]"
            else:
                profit_display = f"[yellow]{total_payout:.4f}[/yellow]"

            # Compact statistics display
            stats_line = f"[cyan]Games: {games_played} | W: {wins} L: {losses} T: {break_evens} | WinRate: {win_rate:.1f}% | Balance: {balance_display} | Total: {profit_display}"

            if args.fibonacci:
                stats_line += f" | FibIndex: {bot.fibonacci_index}/{len(bot.fibonacci_sequence)-1} | MaxFib: {bot.max_fibonacci_reached} | NextBet: {bot.current_bet_amount:.4f}[/cyan]"
            elif args.win_step:
                stats_line += f" | Step: {bot.current_step}/{len(bot.win_step_array)-1} | MaxStep: {bot.max_step_reached} | NextBet: {bot.current_bet_amount:.4f}[/cyan]"
            elif args.martingale:
                stats_line += f" | MaxLossStreak: {max_consecutive_losses} | CurLosses: {bot.consecutive_losses} | NextBet: {bot.current_bet_amount:.4f}[/cyan]"
            else:
                stats_line += "[/cyan]"

            console.print(stats_line)

            # Add a delay between games
            time.sleep(0)
            
        except KeyboardInterrupt:
            console.print("[bold red]🛑 Bot stopped by user[/bold red]")
            break
        except Exception as e:
            console.print(f"[bold red]❌ Error occurred: {str(e)}[/bold red]")
            console.print("[yellow]⏳ Waiting 5 seconds before retry...[/yellow]")
            time.sleep(5)
            console.print()
            continue
    
    # Final statistics
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Calculate hours, minutes, seconds
    total_seconds = int(duration.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    seconds = total_seconds % 60
    
    # Format duration string
    if hours > 0:
        duration_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        duration_str = f"{minutes}m {seconds}s"
    else:
        duration_str = f"{seconds}s"
    
    # Calculate win rate excluding ties
    decided_games = wins + losses
    win_rate = (wins / decided_games * 100) if decided_games > 0 else 0

    stats_text = f"[bold green]Session Statistics:[/bold green]\n"
    stats_text += f"[cyan]Start Time:[/cyan] {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    stats_text += f"[cyan]End Time:[/cyan] {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    stats_text += f"[cyan]Duration:[/cyan] {duration_str}\n"
    stats_text += f"[cyan]Games Played:[/cyan] {games_played}\n"
    stats_text += f"[green]Wins:[/green] {wins}\n"
    stats_text += f"[red]Losses:[/red] {losses}\n"
    stats_text += f"[yellow]Ties:[/yellow] {break_evens}\n"
    stats_text += f"[blue]Win Rate:[/blue] {win_rate:.1f}% (excluding ties)\n"
    if args.fibonacci:
        stats_text += f"[purple]Max Fibonacci Level Reached:[/purple] {bot.max_fibonacci_reached}/{len(bot.fibonacci_sequence)-1}\n"
        stats_text += f"[purple]Current Fibonacci Index:[/purple] {bot.fibonacci_index}\n"
        stats_text += f"[purple]Next Bet Amount:[/purple] {bot.current_bet_amount:.4f} {args.currency.upper()}\n"
    elif args.win_step:
        stats_text += f"[purple]Max Step Reached:[/purple] {bot.max_step_reached}/{len(bot.win_step_array)-1}\n"
        stats_text += f"[purple]Current Step:[/purple] {bot.current_step}/{len(bot.win_step_array)-1}\n"
        stats_text += f"[purple]Current Consecutive Wins:[/purple] {bot.win_step_consecutive_wins}/{bot.win_step_wins}\n"
        stats_text += f"[purple]Next Bet Amount:[/purple] {bot.current_bet_amount:.4f} {args.currency.upper()}\n"
    elif args.martingale:
        stats_text += f"[purple]Max Consecutive Losses (Martingale Level):[/purple] {max_consecutive_losses}\n"
        stats_text += f"[purple]Current Consecutive Losses:[/purple] {bot.consecutive_losses}\n"
        stats_text += f"[purple]Current Consecutive Wins:[/purple] {bot.consecutive_wins}/{args.wins}\n"
        stats_text += f"[purple]Next Bet Amount:[/purple] {bot.current_bet_amount:.4f} {args.currency.upper()}\n"
    
    # Color code total profit/loss
    if total_payout > 0:
        profit_display = f"[bold green]Total Profit: +{total_payout:.4f}[/bold green]"
    elif total_payout < 0:
        profit_display = f"[bold red]Total Loss: {total_payout:.4f}[/bold red]"
    else:
        profit_display = f"[bold yellow]Total: {total_payout:.4f}[/bold yellow]"
    
    stats_text += f"[cyan]Net Result:[/cyan] {profit_display}"
    
    # Add Take Profit information
    if args.take_profit > 0:
        if total_payout >= args.take_profit:
            tp_status = f"[bold green]✅ Target Reached ({total_payout:.4f}/{args.take_profit:.4f})[/bold green]"
        else:
            progress = (total_payout / args.take_profit * 100) if args.take_profit > 0 else 0
            tp_status = f"[yellow]❌ Not Reached ({progress:.1f}% - {total_payout:.4f}/{args.take_profit:.4f})[/yellow]"
        stats_text += f"\n[cyan]Take Profit Status:[/cyan] {tp_status}"
    
    stats_panel = Panel(stats_text, title="[bold blue]📊 Final Stats[/bold blue]", border_style="blue")
    console.print(stats_panel)
    
    # Footer
    footer_text = Text("Thanks for using Baccarat Bot! 🎴", style="italic cyan", justify="center")
    console.print()
    console.print(footer_text)