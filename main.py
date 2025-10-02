import cloudscraper
import json
import random
import argparse
import sys
import yaml
import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich import print as rprint

console = Console()

def load_config(config_file='config.yml'):
    """
    โหลดการตั้งค่าจากไฟล์ YAML
    ถ้าไม่มีไฟล์หรือมีข้อผิดพลาด จะใช้ค่าเริ่มต้น
    """
    default_config = {
        'game': {
            'amount': 0.05,
            'currency': 'trx',
            'mines': 3,
            'shots': 5
        },
        'selection': {
            'mode': 'smart',
            'retry': 3,
            'single_shot_mode': 'win-stay'
        },
        'martingale': {
            'enabled': False,
            'multiplier': 4.0,
            'wins': 4,
            'max_levels': 10
        },
        'win_step': {
            'enabled': False,
            'multiplier': 2.0,
            'wins': 3,
            'array': [1,1,1,1,1,1,1,2,2,2,2,3,3,4,4,5,5,6,7,8,9,11,12,14,16,18,21,24,27,31,36,41,47,54,62,71,81,93,106,121,136,153,172,193,216]
        },
        'auth': {
            'token': ''
        },
        'advanced': {
            'take_profit': 0
        }
    }

    if not os.path.exists(config_file):
        console.print(f"[yellow]⚠️ Config file '{config_file}' not found. Using default values.[/yellow]")
        return default_config

    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        console.print(f"[green]✅ Loaded configuration from '{config_file}'[/green]")

        # Merge กับ default config เพื่อให้แน่ใจว่ามีค่าครบ
        def merge_dict(default, loaded):
            result = default.copy()
            for key, value in loaded.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_dict(result[key], value)
                else:
                    result[key] = value
            return result

        return merge_dict(default_config, config)

    except Exception as e:
        console.print(f"[red]❌ Error loading config file '{config_file}': {str(e)}[/red]")
        console.print("[yellow]⚠️ Using default values instead.[/yellow]")
        return default_config

def load_access_token(config):
    """
    โหลด access token จาก config หรือไฟล์ token.txt
    """
    # ลองอ่านจาก config ก่อน
    if config.get('auth', {}).get('token', '').strip():
        console.print("[green]✅ Using token from config file[/green]")
        return config['auth']['token'].strip()

    # ถ้าไม่มีใน config ให้อ่านจากไฟล์ token.txt
    try:
        with open('token.txt', 'r') as f:
            token = f.read().strip()
        console.print("[green]✅ Using token from token.txt file[/green]")
        return token
    except FileNotFoundError:
        console.print("[bold red]❌ Error: No token found in config and token.txt file not found![/bold red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]❌ Error reading token: {str(e)}[/bold red]")
        sys.exit(1)

class MinesBot:
    def __init__(self, bet_amount=0.1, use_martingale=False, required_wins=4, martingale_multiplier=4, max_martingale=10, max_shots=5, retry_count=3, selection_mode="row", single_shot_mode="win-stay", use_win_step=False, win_step_array=None, win_step_multiplier=2.0, win_step_wins=3, access_token=None):
        self.identifier = self.generate_identifier()
        self.base_bet_amount = bet_amount
        self.current_bet_amount = bet_amount
        self.use_martingale = use_martingale
        self.required_wins = required_wins  # Configurable wins requirement
        self.martingale_multiplier = martingale_multiplier  # Configurable multiplier
        self.max_martingale = max_martingale  # Maximum martingale levels allowed
        self.max_shots = max_shots  # Configurable number of shots
        self.retry_count = retry_count  # How many times to retry same row before random
        self.selection_mode = selection_mode  # "row", "random", or "smart"
        self.single_shot_mode = single_shot_mode  # "fixed", "win-return", or "win-stay"

        # Win Step configuration
        self.use_win_step = use_win_step
        self.win_step_array = win_step_array if win_step_array else [1,1,1,1,1,1,1,2,2,2,2,3,3,4,4,5,5,6,7,8,9,11,12,14,16,18,21,24,27,31,36,41,47,54,62,71,81,93,106,121,136,153,172,193,216]
        self.win_step_multiplier = win_step_multiplier
        self.win_step_wins = win_step_wins
        self.current_step = 0  # Current position in step array
        self.win_step_consecutive_wins = 0  # Track wins for win-step mode

        # Standard Martingale tracking
        self.consecutive_losses = 0
        self.consecutive_wins = 0  # Track consecutive wins
        self.total_losses = 0  # Track total losses for Martingale
        self.scraper = cloudscraper.create_scraper()
        self.used_fields = []  # Track used fields to avoid duplicates
        self.last_lost_row = None  # Track the row where we lost last time
        self.current_retry_attempts = 0  # Track how many times we've retried the same row
        self.single_shot_field = 0  # Track field for single shot mode (0 or 1)

        # Use provided access token
        if not access_token:
            console.print("[bold red]❌ Error: No access token provided![/bold red]")
            sys.exit(1)

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

    def smart_random_selection(self, available_fields, mines_count=3):
        """
        Smart selection algorithm that considers probability and position patterns
        Args:
            available_fields: List of available field numbers (0-24)
            mines_count: Number of mines in the game (default 3)
        Returns:
            Selected field number with highest safety probability
        """
        if not available_fields:
            return None
            
        total_fields = 25
        used_count = len(self.used_fields)
        remaining_safe_fields = total_fields - mines_count - used_count
        
        # Calculate base probability for each field
        base_safety_prob = remaining_safe_fields / len(available_fields) if available_fields else 0
        
        # Apply position-based weighting
        field_weights = {}
        
        for field in available_fields:
            # Convert field number to grid position (0-24 -> row,col)
            row = field // 5
            col = field % 5
            
            # Base weight starts with safety probability
            weight = base_safety_prob
            
            # Pattern-based adjustments:
            # 1. Corner positions (slightly safer - mines tend to avoid corners)
            if (row == 0 or row == 4) and (col == 0 or col == 4):
                weight *= 1.15  # +15% for corners
                
            # 2. Edge positions (moderately safer)
            elif row == 0 or row == 4 or col == 0 or col == 4:
                weight *= 1.08  # +8% for edges
                
            # 3. Center positions (slightly less safe but not much)
            elif row == 2 and col == 2:
                weight *= 0.95  # -5% for exact center
                
            # 4. Avoid clustering around already used fields
            clustering_penalty = 0
            for used_field in self.used_fields:
                used_row = used_field // 5
                used_col = used_field % 5
                distance = abs(row - used_row) + abs(col - used_col)  # Manhattan distance
                
                if distance == 1:  # Adjacent field
                    clustering_penalty += 0.05  # -5% per adjacent used field
                elif distance == 2:  # Near field
                    clustering_penalty += 0.02  # -2% per near used field
                    
            weight *= (1 - min(clustering_penalty, 0.3))  # Cap penalty at 30%
            
            # 5. Diagonal pattern bonus (mines often avoid perfect diagonals)
            if row == col or row + col == 4:  # Main diagonals
                weight *= 1.05  # +5% for diagonal positions
                
            field_weights[field] = max(weight, 0.1)  # Minimum weight of 10%
        
        # Weighted random selection
        fields = list(field_weights.keys())
        weights = list(field_weights.values())
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
            
            # Use weighted random choice
            selected_field = random.choices(fields, weights=weights, k=1)[0]
            
            return selected_field
        else:
            # Fallback to pure random if weights fail
            return random.choice(available_fields)

    def place_bet(self, amount=0, currency="trx", mines_count=3):
        url = "https://stake.com/_api/casino/mines/bet"
        data = {
            "currency": currency,
            "amount": amount,
            "minesCount": mines_count
        }

        try:
            response = self.scraper.post(url, headers=self.headers, json=data)
            result = response.json()
            
            # Reset used fields for new game
            self.used_fields = []
            # Don't reset selected_row here - let next_bet() handle row selection strategy
            
            # Create a beautiful panel for the bet amount response
            bet_info = f"[bold green]Bet Placed Successfully![/bold green]\n"
            bet_info += f"[cyan]Amount:[/cyan] {amount} {currency.upper()}\n"
            bet_info += f"[cyan]Mines Count:[/cyan] {mines_count}\n"
            bet_info += f"[cyan]Game ID:[/cyan] {self.identifier}"
            
            panel = Panel(bet_info, title="[bold blue]💣 Mines Bet[/bold blue]", border_style="blue")
            console.print(panel)
            
            return result
        except Exception as e:
            error_text = Text(f"Error in place_bet: {str(e)}", style="bold red")
            console.print(error_text)
            return None

    def next_bet(self, field_number=None):
        if field_number is None:
            # Special case: If max_shots = 1, alternate between field 0 and 1
            if self.max_shots == 1:
                field_number = self.single_shot_field
                console.print(f"[bold cyan]🎯 Single shot mode: Using field {field_number}[/bold cyan]")
            # Check selection mode
            elif self.selection_mode == "random":
                # Pure random selection - avoid used fields
                available_fields = [i for i in range(25) if i not in self.used_fields]
                
                if not available_fields:
                    console.print("[bold red]❌ All fields exhausted![/bold red]")
                    return None

                field_number = random.choice(available_fields)
                
            elif self.selection_mode == "smart":
                # Smart selection with probability calculation
                available_fields = [i for i in range(25) if i not in self.used_fields]
                
                if not available_fields:
                    console.print("[bold red]❌ All fields exhausted![/bold red]")
                    return None

                field_number = self.smart_random_selection(available_fields)
                
            else:  # selection_mode == "row"
                # Choose row strategy with retry limit
                # Grid is 5x5 (0-24), rows are: 0-4, 5-9, 10-14, 15-19, 20-24
                if not hasattr(self, 'selected_row') or self.selected_row is None:
                    # Define all available rows
                    rows = [
                        [0, 1, 2, 3, 4],      # Row 1
                        [5, 6, 7, 8, 9],      # Row 2  
                        [10, 11, 12, 13, 14], # Row 3
                        [15, 16, 17, 18, 19], # Row 4
                        [20, 21, 22, 23, 24]  # Row 5
                    ]
                    
                    # Strategy: If we lost last time and haven't exceeded retry limit, retry same row
                    if self.last_lost_row is not None and self.current_retry_attempts < self.retry_count:
                        self.selected_row = self.last_lost_row.copy()  # Make a copy
                        self.current_retry_attempts += 1
                    else:
                        # First game, won last time, or exceeded retry limit: randomly select a row
                        self.selected_row = random.choice(rows)
                        self.current_retry_attempts = 0  # Reset retry counter
                        self.last_lost_row = None  # Clear failed row memory
                
                # Find next available field from selected row using smart selection
                available_in_row = [field for field in self.selected_row if field not in self.used_fields]

                if not available_in_row:
                    console.print("[bold red]❌ Selected row exhausted![/bold red]")
                    return None

                # Use smart selection within the row if more than one field available
                if len(available_in_row) > 1:
                    field_number = self.smart_random_selection(available_in_row)
                else:
                    field_number = available_in_row[0]  # Take the only available field
        
        # Add to used fields (except for single shot mode with field 0)
        if not (self.max_shots == 1 and field_number == 0):
            self.used_fields.append(field_number)
        
        url = "https://stake.com/_api/casino/mines/next"
        data = {"fields": [field_number]}

        try:
            response = self.scraper.post(url, headers=self.headers, json=data)
            result = response.json()

            # Check for errors first
            if "errors" in result and result["errors"]:
                error_msg = result["errors"][0].get("message", "Unknown error")
                console.print(f"[bold red]❌ Next bet error: {error_msg}[/bold red]")
                console.print(f"[yellow]Response: {json.dumps(result, indent=2)}[/yellow]")
                # Error is not a loss, just return None to stop the game
                return None

            # # Check if game ended (hit mine)
            # if "minesNext" not in result:
            #     # No minesNext - could be error or game ended
            #     console.print(f"[bold red]💥 Hit a Mine![/bold red]")

            #     # Remember this row for next game (we lost here) - only for row mode
            #     if self.selection_mode == "row" and hasattr(self, 'selected_row') and self.selected_row is not None:
            #         self.last_lost_row = self.selected_row.copy()  # Make a copy

            #     # Clear selected_row so next game will choose strategy in next_bet()
            #     if hasattr(self, 'selected_row'):
            #         self.selected_row = None

            #     # Return a special indicator for game over
            #     return {"game_over": True, "result": "lost", "response": result}

            # Check game status
            mines_next = result["minesNext"]
            is_active = mines_next.get("active", False)

            if is_active:
                # Game is still active - safe field
                if "state" in mines_next and "rounds" in mines_next["state"]:
                    latest_round = mines_next["state"]["rounds"][-1]
                    field = latest_round["field"]
                    payout_multiplier = latest_round["payoutMultiplier"]
                    console.print(f"[green]✅ Field {field} safe! Multiplier: {payout_multiplier:.4f}x[/green]")
            else:
                # Game ended (active: false) - hit mine
                console.print(f"[bold red]💥 Hit a Mine! Game ended[/bold red]")

                # Remember this row for next game (we lost here) - only for row mode
                if self.selection_mode == "row" and hasattr(self, 'selected_row') and self.selected_row is not None:
                    self.last_lost_row = self.selected_row.copy()

                # Clear selected_row so next game will choose strategy in next_bet()
                if hasattr(self, 'selected_row'):
                    self.selected_row = None

                # Return a special indicator for game over
                return {"game_over": True, "result": "lost", "response": result}

            return result
        except Exception as e:
            error_text = Text(f"Error in next_bet: {str(e)}", style="bold red")
            console.print(error_text)
            return None

    def cash_out(self):
        url = "https://stake.com/_api/casino/mines/cashout"
        data = {"identifier": self.identifier}

        try:
            response = self.scraper.post(url, headers=self.headers, json=data)
            result = response.json()

            # Check for errors first
            if "errors" in result and result["errors"]:
                error_msg = result["errors"][0].get("message", "Unknown error")
                console.print(f"[bold red]❌ Cash out error: {error_msg}[/bold red]")
                return None

            # Check if the expected structure exists
            if "minesCashout" not in result:
                console.print(f"[bold red]❌ Error: 'minesCashout' key not found[/bold red]")
                console.print(f"[yellow]Response: {json.dumps(result, indent=2)}[/yellow]")
                return None

            # Create a success panel for cash out
            cash_info = f"[bold green]Cash Out Successful![/bold green]\n"
            cash_info += f"[cyan]Game ID:[/cyan] {self.identifier}\n"
            cash_info += f"[cyan]Status:[/cyan] Completed"

            panel = Panel(cash_info, title="[bold green]💰 Cash Out[/bold green]", border_style="green")
            console.print(panel)

            return result
        except Exception as e:
            error_text = Text(f"Error in cash_out: {str(e)}", style="bold red")
            console.print(error_text)
            return None

    def play_game(self, currency="trx", mines_count=3):
        # Start the game
        bet_result = self.place_bet(amount=self.current_bet_amount, currency=currency, mines_count=mines_count)
        if not bet_result:
            console.print("[bold red]❌ Failed to place bet[/bold red]")
            return -self.current_bet_amount

        # Make configurable number of shots
        shots_made = 0
        game_active = True

        for shot in range(self.max_shots):
            if not game_active:
                break

            result = self.next_bet()
            
            if not result:
                console.print("[bold red]❌ Failed to make shot[/bold red]")
                return -self.current_bet_amount
            
            # Check if game ended (hit mine)
            if isinstance(result, dict) and result.get("game_over"):
                return -self.current_bet_amount

            # Check if game is still active
            if "minesNext" in result:
                game_active = result["minesNext"].get("active", False)
                shots_made += 1

                if not game_active:
                    break
            else:
                console.print("[bold red]❌ Invalid response format[/bold red]")
                return -self.current_bet_amount
        
        if shots_made == 0:
            console.print("[bold red]💀 No successful shots made[/bold red]")
            return -self.current_bet_amount

        # Cash out after successful shots
        data_cashout = self.cash_out()
        
        payout = -self.current_bet_amount  # Default to loss if no cashout data
        if data_cashout and "minesCashout" in data_cashout:
            payout_amount = data_cashout["minesCashout"]["payout"]
            bet_amount = data_cashout["minesCashout"]["amount"]
            
            # If payout is 0, it means we lost the bet
            if payout_amount == 0:
                payout = -bet_amount  # Full loss of bet amount
                result_title = "[bold red]� Game Results - LOST[/bold red]"
                result_style = "red"
            else:
                payout = payout_amount - bet_amount  # Calculate actual profit/loss
                if payout > 0:
                    result_title = "[bold green]🏆 Game Results - WON[/bold green]"
                    result_style = "green"
                else:
                    result_title = "[bold yellow]🏆 Game Results - BREAK EVEN[/bold yellow]"
                    result_style = "yellow"
            
            # Display final results
            payout_info = f"[bold yellow]Final Results:[/bold yellow]\n"
            payout_info += f"[cyan]Shots Made:[/cyan] {shots_made}/{self.max_shots}\n"
            payout_info += f"[cyan]Bet Amount:[/cyan] {bet_amount} {currency.upper()}\n"
            payout_info += f"[cyan]Payout Amount:[/cyan] {payout_amount} {currency.upper()}\n"
            payout_info += f"[cyan]Payout Multiplier:[/cyan] {data_cashout['minesCashout']['payoutMultiplier']}\n"
            
            # Color code profit/loss
            if payout > 0:
                profit_text = f"[bold green]Profit: +{payout}[/bold green]"
            elif payout < 0:
                profit_text = f"[bold red]Loss: {payout}[/bold red]"
            else:
                profit_text = f"[bold yellow]Break Even: {payout}[/bold yellow]"
            
            payout_info += f"[cyan]Net Result:[/cyan] {profit_text}"
            
            result_panel = Panel(payout_info, title=result_title, border_style=result_style)
            console.print(result_panel)
        else:
            # Game ended with loss
            loss_info = f"[bold red]Game Lost![/bold red]\n"
            loss_info += f"[cyan]Bet Amount:[/cyan] {self.current_bet_amount} {currency.upper()}\n"
            loss_info += f"[red]Net Result: -{self.current_bet_amount}[/red]"

            result_panel = Panel(loss_info, title="[bold red]� Game Results[/bold red]", border_style="red")
            console.print(result_panel)
        
        # Return payout and last multiplier for Martingale calculation
        last_multiplier = 1.125  # Default multiplier for mines
        if data_cashout and "minesCashout" in data_cashout:
            last_multiplier = data_cashout["minesCashout"]["payoutMultiplier"]
        
        return payout, last_multiplier

    def update_martingale(self, won=False, last_multiplier=1.075):
        """Update bet amount based on Martingale or Win Step strategy"""
        # Handle single shot field switching (works with any strategy)
        if self.max_shots == 1:
            if self.single_shot_mode == "fixed":
                # Mode 1: Always use field 0
                self.single_shot_field = 0
            elif self.single_shot_mode == "win-return":
                # Mode 2: Loss→switch, Win→return to 0
                if won:
                    self.single_shot_field = 0
                    console.print(f"[bold green]✅ Win! Return to field {self.single_shot_field}[/bold green]")
                else:
                    self.single_shot_field = 11
                    console.print(f"[bold yellow]⚠️ Loss! Switch to field {self.single_shot_field}[/bold yellow]")
            elif self.single_shot_mode == "win-stay":
                # Mode 3: Loss→switch, Win→stay
                if won:
                    console.print(f"[bold green]✅ Win! Stay at field {self.single_shot_field}[/bold green]")
                else:
                    self.single_shot_field = 1 if self.single_shot_field == 0 else 0
                    console.print(f"[bold yellow]⚠️ Loss! Switch to field {self.single_shot_field}[/bold yellow]")
            elif self.single_shot_mode == "loss-switch":
                # Mode 4: Loss→switch, Win at field 1→return to 0, Win at field 0→stay
                if won:
                    if self.single_shot_field == 12:
                        self.single_shot_field = 0
                        console.print(f"[bold green]✅ Win at field 1! Return to field 0[/bold green]")
                    else:
                        console.print(f"[bold green]✅ Win at field 0! Stay at field 0[/bold green]")
                else:
                    self.single_shot_field = 12 if self.single_shot_field == 0 else 0
                    console.print(f"[bold yellow]⚠️ Loss! Switch to field {self.single_shot_field}[/bold yellow]")

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
                    console.print(f"[bold green]🎯 {self.win_step_wins} Wins! Reset to step 0 | Bet: {self.current_bet_amount:.4f}[/bold green]")
                else:
                    # Continue martingale on wins
                    self.current_bet_amount *= self.win_step_multiplier
                    console.print(f"[bold green]✅ Win {self.win_step_consecutive_wins}/{self.win_step_wins} | Next bet: {self.current_bet_amount:.4f} (x{self.win_step_multiplier})[/bold green]")
            else:
                # Loss: reset win counter and move to next step
                self.win_step_consecutive_wins = 0

                # Move to next step in array
                if self.current_step < len(self.win_step_array) - 1:
                    self.current_step += 1
                else:
                    # Already at max step, stay there
                    console.print(f"[bold red]⚠️ Max step reached ({self.current_step})![/bold red]")

                self.current_bet_amount = self.base_bet_amount * self.win_step_array[self.current_step]
                console.print(f"[bold red]📉 Loss! Step {self.current_step} | Bet: {self.current_bet_amount:.4f}[/bold red]")

            return

        # Standard Martingale Strategy
        if not self.use_martingale:
            return

        if won:
            # If we won, clear the last lost row and reset retry counter
            if self.last_lost_row is not None:
                self.last_lost_row = None
                self.current_retry_attempts = 0

            # Clear selected_row so next game will choose new strategy
            self.selected_row = None

            # Standard Martingale: Win = reset to base immediately
            self.consecutive_losses = 0
            self.total_losses = 0
            self.current_bet_amount = self.base_bet_amount

            # Track consecutive wins
            self.consecutive_wins += 1
            console.print(f"[bold green]✅ Win! Reset to base bet {self.base_bet_amount:.4f} | Wins: {self.consecutive_wins}[/bold green]")
        else:
            # Reset wins counter when lost
            self.consecutive_wins = 0

            # Add current loss to total
            self.total_losses += self.current_bet_amount
            self.consecutive_losses += 1

            # Check if we've reached max martingale levels
            if self.consecutive_losses >= self.max_martingale:
                console.print(f"[bold red]🛑 Max Level ({self.max_martingale})! Reset to base[/bold red]")

                # Reset martingale system
                self.consecutive_losses = 0
                self.consecutive_wins = 0
                self.total_losses = 0
                self.current_bet_amount = self.base_bet_amount
            else:
                # Configurable Martingale: Use custom multiplier
                self.current_bet_amount = self.base_bet_amount * (self.martingale_multiplier ** self.consecutive_losses)

                console.print(f"[bold red]📈 Next bet: {self.current_bet_amount:.4f} | Loss {self.consecutive_losses}/{self.max_martingale}[/bold red]")

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
                    console.print(f"[bold green]💰 {currency.upper()} Balance: {target_balance}[/bold green]")
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
    parser = argparse.ArgumentParser(description='Mines Bot for Stake.com')
    parser.add_argument('--amount', type=float, default=0.05, 
                        help='Bet amount (default: 0.05)')
    parser.add_argument('--currency', type=str, default='trx', 
                        help='Currency to use (default: trx). Options: usdt, trx, doge, ltc, etc.')
    parser.add_argument('--martingale', action='store_true',
                        help='Enable Martingale strategy')
    parser.add_argument('--wins', type=int, default=4,
                        help='Number of consecutive wins required to reset Martingale (default: 4)')
    parser.add_argument('--multiplier', type=float, default=4.0,
                        help='Martingale multiplier on loss (default: 4.0)')
    parser.add_argument('--max-martingale', type=int, default=10,
                        help='Maximum martingale levels before reset (default: 10)')
    parser.add_argument('--mines', type=int, default=3,
                        help='Number of mines in the game (default: 3)')
    parser.add_argument('--shots', type=int, default=5,
                        help='Number of shots to make per game (default: 5)')
    parser.add_argument('--retry', type=int, default=3,
                        help='Number of times to retry same row before switching (default: 3)')
    parser.add_argument('--mode', type=str, default='smart', choices=['row', 'random', 'smart'],
                        help='Field selection mode: row (row-based), random (pure random), or smart (probability-based) (default: smart)')
    parser.add_argument('--single-shot-mode', type=str, default='win-stay', choices=['fixed', 'win-return', 'win-stay', 'loss-switch'],
                        help='Single shot field switching mode (only for --shots 1): fixed (always field 0), win-return (loss→switch, win→return to 0), win-stay (loss→switch, win→stay), loss-switch (loss→switch, win at 1→return to 0, win at 0→stay) (default: win-stay)')
    parser.add_argument('--win-step', action='store_true',
                        help='Enable Win Step strategy (alternative to Martingale)')
    parser.add_argument('--win-step-multiplier', type=float, default=2.0,
                        help='Win Step: Multiplier on consecutive wins (default: 2.0)')
    parser.add_argument('--win-step-wins', type=int, default=3,
                        help='Win Step: Number of consecutive wins to reset to step 0 (default: 3)')
    parser.add_argument('--take-profit', type=float, default=0,
                        help='Take profit target - stop bot when profit reaches this amount (default: 0 = disabled)')
    parser.add_argument('--config', type=str, default='config.yml',
                        help='Configuration file path (default: config.yml)')

    args = parser.parse_args()

    # โหลดค่าการตั้งค่าจากไฟล์ YAML
    config = load_config(args.config)

    # ถ้าไม่ได้ระบุค่าใน command line arguments ให้ใช้ค่าจาก config file
    if not any(vars(args).values()):
        # ไม่มี arguments เลย ใช้ค่าจาก config ทั้งหมด
        args.amount = config['game']['amount']
        args.currency = config['game']['currency']
        args.mines = config['game']['mines']
        args.shots = config['game']['shots']
        args.mode = config['selection']['mode']
        args.retry = config['selection']['retry']
        args.single_shot_mode = config['selection']['single_shot_mode']
        args.martingale = config['martingale']['enabled']
        args.multiplier = config['martingale']['multiplier']
        args.wins = config['martingale']['wins']
        args.max_martingale = config['martingale']['max_levels']
        args.win_step = config['win_step']['enabled']
        args.win_step_multiplier = config['win_step']['multiplier']
        args.win_step_wins = config['win_step']['wins']
        args.take_profit = config['advanced']['take_profit']
    else:
        # มี arguments บางส่วน ใช้ค่าจาก config สำหรับตัวที่เป็น default
        if args.amount == 0.05:  # ค่า default
            args.amount = config['game']['amount']
        if args.currency == 'trx':  # ค่า default
            args.currency = config['game']['currency']
        if args.mines == 3:  # ค่า default
            args.mines = config['game']['mines']
        if args.shots == 5:  # ค่า default
            args.shots = config['game']['shots']
        if args.mode == 'smart':  # ค่า default
            args.mode = config['selection']['mode']
        if args.retry == 3:  # ค่า default
            args.retry = config['selection']['retry']
        if args.single_shot_mode == 'win-stay':  # ค่า default
            args.single_shot_mode = config['selection']['single_shot_mode']
        if not args.martingale:  # ค่า default คือ False
            args.martingale = config['martingale']['enabled']
        if args.multiplier == 4.0:  # ค่า default
            args.multiplier = config['martingale']['multiplier']
        if args.wins == 4:  # ค่า default
            args.wins = config['martingale']['wins']
        if args.max_martingale == 10:  # ค่า default
            args.max_martingale = config['martingale']['max_levels']
        if not args.win_step:  # ค่า default คือ False
            args.win_step = config['win_step']['enabled']
        if args.win_step_multiplier == 2.0:  # ค่า default
            args.win_step_multiplier = config['win_step']['multiplier']
        if args.win_step_wins == 3:  # ค่า default
            args.win_step_wins = config['win_step']['wins']
        if args.take_profit == 0:  # ค่า default
            args.take_profit = config['advanced']['take_profit']

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
    
    # Validate mines count
    if args.mines < 1 or args.mines > 24:
        console.print("[bold red]❌ Error: Mines count must be between 1 and 24[/bold red]")
        sys.exit(1)
    
    # Validate shots count
    if args.shots < 1 or args.shots > 20:
        console.print("[bold red]❌ Error: Shots count must be between 1 and 20[/bold red]")
        sys.exit(1)
    
    # Validate retry count
    if args.retry < 1 or args.retry > 10:
        console.print("[bold red]❌ Error: Retry count must be between 1 and 10[/bold red]")
        sys.exit(1)
    
    # Clear screen and show header
    console.clear()
    header_text = Text("💣 STAKE.COM MINES BOT - CONTINUOUS MODE 💣", style="bold white on blue", justify="center")
    console.print(header_text)
    
    # Display configuration
    config_info = f"[cyan]Config File:[/cyan] {args.config}\n"
    config_info += f"[cyan]Base Bet Amount:[/cyan] {args.amount} {args.currency.upper()}\n"
    config_info += f"[cyan]Currency:[/cyan] {args.currency.upper()}\n"
    config_info += f"[cyan]Mines Count:[/cyan] {args.mines}\n"
    config_info += f"[cyan]Shots Per Game:[/cyan] {args.shots}\n"
    if args.shots == 1:
        config_info += f"[cyan]Single Shot Mode:[/cyan] {args.single_shot_mode}\n"
    config_info += f"[cyan]Selection Mode:[/cyan] {args.mode.title()}\n"
    if args.mode == "row":
        config_info += f"[cyan]Row Retry Count:[/cyan] {args.retry}\n"
    elif args.mode == "smart":
        config_info += f"[cyan]Smart Algorithm:[/cyan] Probability-based with pattern analysis\n"
    if args.take_profit > 0:
        config_info += f"[cyan]Take Profit Target:[/cyan] {args.take_profit:.4f} {args.currency.upper()}\n"
    config_info += f"[cyan]Mode:[/cyan] Continuous\n"

    # Betting strategy info
    if args.win_step:
        config_info += f"[cyan]Strategy:[/cyan] Win Step"
        config_info += f"\n[yellow]⚠️  Win Step Strategy:[/yellow]"
        config_info += f"\n[yellow]   • Win multiplier: {args.win_step_multiplier}x[/yellow]"
        config_info += f"\n[yellow]   • Reset after {args.win_step_wins} consecutive wins[/yellow]"
        config_info += f"\n[yellow]   • Step array: {len([1,1,1,1,1,1,1,2,2,2,2,3,3,4,4,5,5,6,7,8,9,11,12,14,16,18,21,24,27,31,36,41,47,54,62,71,81,93,106,121,136,153,172,193,216])} steps[/yellow]"
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
    max_consecutive_losses = 0  # Track maximum Martingale level
    
    # Load access token
    access_token = load_access_token(config)

    # Create bot instance with betting strategy settings
    bot = MinesBot(bet_amount=args.amount, use_martingale=args.martingale,
                   required_wins=args.wins, martingale_multiplier=args.multiplier,
                   max_martingale=args.max_martingale,
                   max_shots=args.shots, retry_count=args.retry, selection_mode=args.mode,
                   single_shot_mode=args.single_shot_mode,
                   use_win_step=args.win_step, win_step_array=config['win_step']['array'],
                   win_step_multiplier=args.win_step_multiplier,
                   win_step_wins=args.win_step_wins, access_token=access_token)
    
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
            result = bot.play_game(currency=args.currency, mines_count=args.mines)
            
            # Handle return value (payout, multiplier)
            if isinstance(result, tuple):
                payout, last_multiplier = result
            else:
                payout = result
                last_multiplier = 1.125  # Default mines multiplier
            
            total_payout += payout
            
            # Update Martingale strategy based on result
            won = payout > 0
            bot.update_martingale(won=won, last_multiplier=last_multiplier)
            
            # Track maximum consecutive losses for Martingale
            max_consecutive_losses = max(max_consecutive_losses, bot.consecutive_losses)
            
            # Track win/loss statistics
            if payout > 0:
                wins += 1
            elif payout < 0:
                losses += 1
            else:
                break_evens += 1
            
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
            
            # Show real-time statistics after each game
            win_rate = (wins / games_played * 100) if games_played > 0 else 0
            
            # Get current balance for live stats
            current_balance = bot.get_balance(args.currency)
            balance_display = f"{current_balance:.4f} {args.currency.upper()}" if current_balance is not None else "N/A"
            
            realtime_stats = f"[bold cyan]📊 Current Session Stats:[/bold cyan]\n"
            realtime_stats += f"[cyan]Games:[/cyan] {games_played} | "
            realtime_stats += f"[green]Wins:[/green] {wins} | "
            realtime_stats += f"[red]Losses:[/red] {losses} | "
            realtime_stats += f"[yellow]Break Evens:[/yellow] {break_evens}\n"
            realtime_stats += f"[blue]Win Rate:[/blue] {win_rate:.1f}% | "
            realtime_stats += f"[magenta]Balance:[/magenta] {balance_display}\n"
            
            # Color code total profit/loss
            if total_payout > 0:
                profit_display = f"[bold green]Profit: +{total_payout:.4f}[/bold green]"
            elif total_payout < 0:
                profit_display = f"[bold red]Loss: {total_payout:.4f}[/bold red]"
            else:
                profit_display = f"[bold yellow]Even: {total_payout:.4f}[/bold yellow]"
            
            realtime_stats += f"[cyan]Total:[/cyan] {profit_display}"
            
            # Add Take Profit progress if enabled
            if args.take_profit > 0:
                progress_percentage = (total_payout / args.take_profit * 100) if args.take_profit > 0 else 0
                progress_display = f"[magenta]Take Profit Progress:[/magenta] {progress_percentage:.1f}% ({total_payout:.4f}/{args.take_profit:.4f})"
                realtime_stats += f"\n{progress_display}"
            
            if args.win_step:
                realtime_stats += f"\n[purple]Current Step:[/purple] {bot.current_step}/{len(bot.win_step_array)-1} | "
                realtime_stats += f"[purple]Consecutive Wins:[/purple] {bot.win_step_consecutive_wins}/{bot.win_step_wins} | "
                realtime_stats += f"[purple]Next Bet:[/purple] {bot.current_bet_amount:.4f} {args.currency.upper()}"
            elif args.martingale:
                realtime_stats += f"\n[purple]Consecutive Losses:[/purple] {bot.consecutive_losses} | "
                realtime_stats += f"[purple]Max Losses:[/purple] {max_consecutive_losses} | "
                realtime_stats += f"[purple]Next Bet:[/purple] {bot.current_bet_amount:.4f} {args.currency.upper()}"
            
            stats_panel = Panel(realtime_stats, title="[bold blue]📈 Live Statistics[/bold blue]", border_style="blue")
            console.print(stats_panel)

            # Add a delay between games
            time.sleep(0)
            console.print()
            
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
    
    win_rate = (wins / games_played * 100) if games_played > 0 else 0
    
    stats_text = f"[bold green]Session Statistics:[/bold green]\n"
    stats_text += f"[cyan]Start Time:[/cyan] {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    stats_text += f"[cyan]End Time:[/cyan] {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
    stats_text += f"[cyan]Duration:[/cyan] {duration_str}\n"
    stats_text += f"[cyan]Games Played:[/cyan] {games_played}\n"
    stats_text += f"[green]Wins:[/green] {wins}\n"
    stats_text += f"[red]Losses:[/red] {losses}\n"
    stats_text += f"[yellow]Break Evens:[/yellow] {break_evens}\n"
    stats_text += f"[blue]Win Rate:[/blue] {win_rate:.1f}%\n"
    if args.martingale:
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
    footer_text = Text("Thanks for using Mines Bot! 💣", style="italic cyan", justify="center")
    console.print()
    console.print(footer_text)