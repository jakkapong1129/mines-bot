import sys
import os
import yaml
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import time
from datetime import datetime

# Add parent directory to path to import bot classes
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

# Import bot modules
import importlib.util

# Load BaccaratBot
baccarat_path = os.path.join(parent_dir, 'baccarat', 'main-dev.py')
spec = importlib.util.spec_from_file_location("baccarat_module", baccarat_path)
baccarat_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(baccarat_module)
BaccaratBot = baccarat_module.BaccaratBot

# Load RouletteBot
roulette_path = os.path.join(parent_dir, 'roulette', 'main-dev.py')
spec = importlib.util.spec_from_file_location("roulette_module", roulette_path)
roulette_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(roulette_module)
RouletteBot = roulette_module.RouletteBot

console = Console()

class HybridBot:
    def __init__(self, access_token, config):
        self.access_token = access_token
        self.config = config
        self.current_game = "baccarat"  # Start with baccarat
        self.switch_wins = config.get('switch_wins', 0)  # 0 = disabled
        self.switch_loss = config.get('switch_loss', 0)  # 0 = disabled
        self.total_wins = 0  # Total wins (not consecutive)
        self.switch_consecutive_losses = 0  # Consecutive losses for game switching (resets on switch)
        self.martingale_consecutive_losses = 0  # Consecutive losses for martingale (continues across games)

        # Create bot instances
        self.baccarat_bot = None
        self.roulette_bot = None

        # Initialize bots
        self._init_baccarat_bot()
        self._init_roulette_bot()

    def _init_baccarat_bot(self):
        """Initialize Baccarat bot with config"""
        self.baccarat_bot = BaccaratBot(
            access_token=self.access_token,
            bet_amount=self.config.get('amount', 0.003),
            use_martingale=self.config.get('martingale', False),
            required_wins=self.config.get('wins', 1),
            martingale_multiplier=self.config.get('multiplier', 2.0),
            max_martingale=self.config.get('max_martingale', 10),
            use_win_step=self.config.get('win_step', False),
            win_step_array=self.config.get('win_step_array', None),
            win_step_multiplier=self.config.get('win_step_multiplier', 2.0),
            win_step_wins=self.config.get('win_step_wins', 3),
            use_fibonacci=self.config.get('fibonacci', False),
            max_fibonacci=self.config.get('max_fibonacci', 15),
            switch_color_after_losses=self.config.get('baccarat_switch_after_losses', 0),
            switch_bet_count=self.config.get('baccarat_switch_bet_count', 0),
            base_color=self.config.get('baccarat_color', 'player')
        )

    def _init_roulette_bot(self):
        """Initialize Roulette bot with config"""
        self.roulette_bot = RouletteBot(
            access_token=self.access_token,
            bet_amount=self.config.get('amount', 0.003),
            use_martingale=self.config.get('martingale', False),
            required_wins=self.config.get('wins', 1),
            martingale_multiplier=self.config.get('multiplier', 2.0),
            max_martingale=self.config.get('max_martingale', 10),
            use_win_step=self.config.get('win_step', False),
            win_step_array=self.config.get('win_step_array', None),
            win_step_multiplier=self.config.get('win_step_multiplier', 2.0),
            win_step_wins=self.config.get('win_step_wins', 3),
            switch_color_after_losses=self.config.get('roulette_switch_after_losses', 0),
            switch_bet_count=self.config.get('roulette_switch_bet_count', 0),
            base_color=self.config.get('roulette_color', 'colorRed')
        )

    def get_current_bot(self):
        """Get the currently active bot"""
        if self.current_game == "baccarat":
            return self.baccarat_bot
        else:
            return self.roulette_bot

    def switch_game(self, reason=""):
        """Switch between games"""
        if self.current_game == "baccarat":
            self.current_game = "roulette"
            console.print(f"\n[bold magenta]🔄 GAME SWITCH: Baccarat → Roulette {reason}[/bold magenta]\n")
        else:
            self.current_game = "baccarat"
            console.print(f"\n[bold magenta]🔄 GAME SWITCH: Roulette → Baccarat {reason}[/bold magenta]\n")

        # Reset counters after switch
        self.total_wins = 0
        self.switch_consecutive_losses = 0  # Reset switch counter (so need X losses again to switch back)

    def play_game(self, currency="usdt"):
        """Play one round of the current game"""
        bot = self.get_current_bot()

        # Sync martingale state to bot BEFORE playing
        bot.consecutive_losses = self.martingale_consecutive_losses

        # Recalculate bet amount based on consecutive losses
        if bot.consecutive_losses > 0:
            if bot.use_martingale:
                bot.current_bet_amount = bot.base_bet_amount * (bot.martingale_multiplier ** bot.consecutive_losses)
            elif bot.use_fibonacci:
                # For fibonacci, we need to track the index separately
                pass  # Fibonacci is more complex, keeping current implementation
            elif bot.use_win_step:
                # For win step, we need to track the step separately
                pass  # Win step is more complex, keeping current implementation
        else:
            bot.current_bet_amount = bot.base_bet_amount

        # Display current game
        game_name = "🎴 BACCARAT" if self.current_game == "baccarat" else "🎰 ROULETTE"
        # Show bet amount BEFORE playing (for accurate display)
        console.print(f"[bold cyan]{game_name} (Bet: {bot.current_bet_amount:.4f})[/bold cyan]", end=" ")

        # Play game
        result = bot.play_game(currency=currency)

        # Handle return value
        if isinstance(result, tuple):
            payout, last_multiplier = result
        else:
            payout = result
            last_multiplier = 2.0

        # Track wins/losses
        if payout > 0:
            self.total_wins += 1
            self.switch_consecutive_losses = 0  # Reset switch counter on win
            self.martingale_consecutive_losses = 0  # Reset martingale counter on win
            bot.update_martingale(won=True, last_multiplier=last_multiplier)

            # Check if we should switch games based on wins
            if self.switch_wins > 0 and self.total_wins >= self.switch_wins:
                self.switch_game(f"(After {self.switch_wins} wins)")
        elif payout < 0:
            self.switch_consecutive_losses += 1  # Increment switch counter
            self.martingale_consecutive_losses += 1  # Increment martingale counter
            bot.update_martingale(won=False, last_multiplier=last_multiplier)

            # Check if we should switch games based on consecutive losses
            if self.switch_loss > 0 and self.switch_consecutive_losses >= self.switch_loss:
                self.switch_game(f"(After {self.switch_loss} consecutive losses)")
        else:
            # Tie in baccarat - don't update counters
            pass

        # Sync martingale state back from bot (including bet amount)
        self.martingale_consecutive_losses = bot.consecutive_losses

        return payout, last_multiplier

    def get_balance(self, currency="usdt"):
        """Get balance using current bot"""
        bot = self.get_current_bot()
        return bot.get_balance(currency)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Hybrid Bot - Switches between Baccarat and Roulette')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Config file path (default: config.yml)')
    parser.add_argument('--amount', type=float, default=None,
                        help='Bet amount (overrides config)')
    parser.add_argument('--currency', type=str, default=None,
                        help='Currency to use (overrides config)')
    parser.add_argument('--switch-wins', type=int, default=None,
                        help='Total wins before switching games (0=disabled, overrides config)')
    parser.add_argument('--switch-loss', type=int, default=None,
                        help='Consecutive losses before switching games (0=disabled, overrides config)')
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
    if args.amount is not None:
        config['amount'] = args.amount
    if args.currency is not None:
        config['currency'] = args.currency
    if args.switch_wins is not None:
        config['switch_wins'] = args.switch_wins
    if args.switch_loss is not None:
        config['switch_loss'] = args.switch_loss
    if args.take_profit is not None:
        config['take_profit'] = args.take_profit

    # Get final values
    amount = config.get('amount', 0.003)
    currency = config.get('currency', 'trx').lower()
    switch_wins = config.get('switch_wins', 0)
    switch_loss = config.get('switch_loss', 0)
    take_profit = config.get('take_profit', 0)

    # Validate amount
    if amount <= 0:
        console.print("[bold red]❌ Error: Amount must be greater than 0[/bold red]")
        sys.exit(1)

    # Validate currency
    valid_currencies = ['usdt', 'trx', 'doge', 'ltc', 'btc', 'eth', 'bch', 'xrp', 'eos', 'bnb']
    if currency not in valid_currencies:
        console.print(f"[bold red]❌ Error: Unsupported currency '{currency}'. Supported currencies: {', '.join(valid_currencies)}[/bold red]")
        sys.exit(1)

    # Clear screen and show header
    console.clear()
    header_text = Text("🎮 HYBRID BOT - BACCARAT ⇄ ROULETTE 🎮", style="bold white on purple", justify="center")
    console.print(header_text)

    # Display configuration
    config_info = f"[cyan]Base Bet Amount:[/cyan] {amount} {currency.upper()}\n"
    config_info += f"[cyan]Currency:[/cyan] {currency.upper()}\n"

    # Game switching info
    if switch_wins > 0 and switch_loss > 0:
        config_info += f"[cyan]Switch After:[/cyan] {switch_wins} wins OR {switch_loss} consecutive losses\n"
    elif switch_wins > 0:
        config_info += f"[cyan]Switch After:[/cyan] {switch_wins} wins\n"
    elif switch_loss > 0:
        config_info += f"[cyan]Switch After:[/cyan] {switch_loss} consecutive losses\n"
    else:
        config_info += f"[cyan]Game Switching:[/cyan] Disabled (play Baccarat only)\n"

    config_info += f"[cyan]Starting Game:[/cyan] Baccarat 🎴\n"
    config_info += f"[cyan]Baccarat Side:[/cyan] {config.get('baccarat_color', 'player').capitalize()}\n"
    config_info += f"[cyan]Roulette Color:[/cyan] {config.get('roulette_color', 'colorRed')}\n"

    if take_profit > 0:
        config_info += f"[cyan]Take Profit Target:[/cyan] {take_profit:.4f} {currency.upper()}\n"

    config_info += f"[cyan]Mode:[/cyan] Continuous\n"

    # Betting strategy info
    if config.get('fibonacci', False):
        config_info += f"[cyan]Strategy:[/cyan] Fibonacci\n"
        config_info += f"[yellow]⚠️  Max levels: {config.get('max_fibonacci', 15)}[/yellow]"
    elif config.get('win_step', False):
        config_info += f"[cyan]Strategy:[/cyan] Win Step\n"
        config_info += f"[yellow]⚠️  Multiplier: {config.get('win_step_multiplier', 2.0)}x, Reset after {config.get('win_step_wins', 3)} wins[/yellow]"
    elif config.get('martingale', False):
        config_info += f"[cyan]Strategy:[/cyan] Martingale\n"
        config_info += f"[yellow]⚠️  Multiplier: {config.get('multiplier', 2.0)}x, Max levels: {config.get('max_martingale', 10)}[/yellow]"
    else:
        config_info += f"[cyan]Strategy:[/cyan] Flat Bet (No progression)"

    config_panel = Panel(config_info, title="[bold yellow]⚙️ Configuration[/bold yellow]", border_style="yellow")
    console.print(config_panel)
    console.print()

    # Record start time
    start_time = datetime.now()

    # Initialize statistics
    games_played = 0
    total_payout = 0
    wins = 0
    losses = 0
    ties = 0
    baccarat_games = 0
    roulette_games = 0
    game_switches = 0
    max_consecutive_losses = 0

    # Create hybrid bot instance
    hybrid_bot = HybridBot(access_token=access_token, config=config)

    # Check balance before starting
    console.print(f"[bold cyan]💰 Checking {currency.upper()} balance...[/bold cyan]")
    current_balance = hybrid_bot.get_balance(currency)

    if current_balance is None:
        console.print("[bold red]❌ Failed to get balance. Exiting...[/bold red]")
        sys.exit(1)

    # Check if we have enough balance for the bet
    if current_balance < amount:
        console.print(f"[bold red]❌ Insufficient balance! Current: {current_balance} {currency.upper()}, Required: {amount} {currency.upper()}[/bold red]")
        sys.exit(1)

    console.print(f"[bold green]✅ Sufficient balance available![/bold green]")
    console.print()

    # Main game loop
    while True:
        try:
            # Check balance before each game
            current_balance = hybrid_bot.get_balance(currency)
            if current_balance is not None:
                current_bot = hybrid_bot.get_current_bot()
                if current_balance < current_bot.current_bet_amount:
                    console.print()
                    console.print(f"[bold red]💰 Balance Check Failed![/bold red]")
                    console.print(f"[bold red]⚠️  Insufficient balance for next bet![/bold red]")
                    console.print(f"[bold red]Current: {current_balance:.4f} {currency.upper()}, Required: {current_bot.current_bet_amount:.4f} {currency.upper()}[/bold red]")
                    console.print("[bold yellow]🛑 Stopping bot due to insufficient balance...[/bold yellow]")
                    break

            # Track current game before playing
            game_before = hybrid_bot.current_game

            # Play game
            payout, last_multiplier = hybrid_bot.play_game(currency=currency)

            # Track game statistics
            if game_before == "baccarat":
                baccarat_games += 1
            else:
                roulette_games += 1

            # Check if game switched
            if game_before != hybrid_bot.current_game:
                game_switches += 1

            total_payout += payout

            # Track win/loss statistics
            if payout > 0:
                wins += 1
            elif payout < 0:
                losses += 1
            else:
                ties += 1

            # Track maximum consecutive losses (from martingale)
            max_consecutive_losses = max(max_consecutive_losses, hybrid_bot.martingale_consecutive_losses)

            games_played += 1

            # Check Take Profit target
            if take_profit > 0 and total_payout >= take_profit:
                console.print()
                console.print(f"[bold green]🎯 Take Profit Target Reached![/bold green]")
                console.print(f"[bold green]Target: {take_profit:.4f} {currency.upper()}, Current Profit: {total_payout:.4f} {currency.upper()}[/bold green]")
                console.print("[bold yellow]🛑 Stopping bot due to Take Profit target reached...[/bold yellow]")
                break

            # Show statistics after each game
            decided_games = wins + losses
            win_rate = (wins / decided_games * 100) if decided_games > 0 else 0

            # Get current balance
            current_balance = hybrid_bot.get_balance(currency)
            balance_display = f"{current_balance:.4f}" if current_balance is not None else "N/A"

            # Color code total profit/loss
            if total_payout > 0:
                profit_display = f"[green]+{total_payout:.4f}[/green]"
            elif total_payout < 0:
                profit_display = f"[red]{total_payout:.4f}[/red]"
            else:
                profit_display = f"[yellow]{total_payout:.4f}[/yellow]"

            # Calculate next bet amount based on current martingale state
            current_bot = hybrid_bot.get_current_bot()
            if hybrid_bot.martingale_consecutive_losses > 0:
                if current_bot.use_martingale:
                    next_bet = current_bot.base_bet_amount * (current_bot.martingale_multiplier ** hybrid_bot.martingale_consecutive_losses)
                elif current_bot.use_fibonacci:
                    next_bet = current_bot.current_bet_amount  # Fibonacci is complex, use current
                elif current_bot.use_win_step:
                    next_bet = current_bot.current_bet_amount  # Win step is complex, use current
                else:
                    next_bet = current_bot.base_bet_amount
            else:
                next_bet = current_bot.base_bet_amount

            # Compact statistics display
            stats_line = f"[cyan]Games: {games_played} (🎴{baccarat_games}/🎰{roulette_games}) | Switch: {game_switches} | "
            stats_line += f"W: {wins} L: {losses} T: {ties} | WinRate: {win_rate:.1f}% | "

            # Show win/loss progress based on enabled switches
            if switch_wins > 0 and switch_loss > 0:
                stats_line += f"W:{hybrid_bot.total_wins}/{switch_wins} SwitchL:{hybrid_bot.switch_consecutive_losses}/{switch_loss} | "
            elif switch_wins > 0:
                stats_line += f"Wins: {hybrid_bot.total_wins}/{switch_wins} | "
            elif switch_loss > 0:
                stats_line += f"SwitchL: {hybrid_bot.switch_consecutive_losses}/{switch_loss} | "

            # Always show martingale state
            stats_line += f"MartL: {hybrid_bot.martingale_consecutive_losses} | "

            stats_line += f"Balance: {balance_display} | Total: {profit_display} | NextBet: {next_bet:.4f}[/cyan]"

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
    stats_text += f"[cyan]Total Games:[/cyan] {games_played}\n"
    stats_text += f"[cyan]Baccarat Games:[/cyan] 🎴 {baccarat_games}\n"
    stats_text += f"[cyan]Roulette Games:[/cyan] 🎰 {roulette_games}\n"
    stats_text += f"[cyan]Game Switches:[/cyan] 🔄 {game_switches}\n"
    stats_text += f"[green]Wins:[/green] {wins}\n"
    stats_text += f"[red]Losses:[/red] {losses}\n"
    stats_text += f"[yellow]Ties:[/yellow] {ties}\n"
    stats_text += f"[blue]Win Rate:[/blue] {win_rate:.1f}% (excluding ties)\n"
    stats_text += f"[purple]Max Consecutive Losses:[/purple] {max_consecutive_losses}\n"

    # Color code total profit/loss
    if total_payout > 0:
        profit_display = f"[bold green]Total Profit: +{total_payout:.4f}[/bold green]"
    elif total_payout < 0:
        profit_display = f"[bold red]Total Loss: {total_payout:.4f}[/bold red]"
    else:
        profit_display = f"[bold yellow]Total: {total_payout:.4f}[/bold yellow]"

    stats_text += f"[cyan]Net Result:[/cyan] {profit_display}"

    # Add Take Profit information
    if take_profit > 0:
        if total_payout >= take_profit:
            tp_status = f"[bold green]✅ Target Reached ({total_payout:.4f}/{take_profit:.4f})[/bold green]"
        else:
            progress = (total_payout / take_profit * 100) if take_profit > 0 else 0
            tp_status = f"[yellow]❌ Not Reached ({progress:.1f}% - {total_payout:.4f}/{take_profit:.4f})[/yellow]"
        stats_text += f"\n[cyan]Take Profit Status:[/cyan] {tp_status}"

    stats_panel = Panel(stats_text, title="[bold blue]📊 Final Stats[/bold blue]", border_style="blue")
    console.print(stats_panel)

    # Footer
    footer_text = Text("Thanks for using Hybrid Bot! 🎮", style="italic cyan", justify="center")
    console.print()
    console.print(footer_text)
