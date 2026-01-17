"""
Qwen Token Extractor - GUI Version
Extract and manage Qwen account tokens from curl commands.
"""

import re
import json
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path


class TokenExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔑 Qwen Token Extractor")
        self.root.geometry("700x600")
        self.root.configure(bg="#1a1a2e")

        self.config_path = Path(__file__).parent / "config.json"

        self.setup_styles()
        self.create_widgets()
        self.load_accounts()

    def setup_styles(self):
        """Setup custom styles."""
        style = ttk.Style()
        style.theme_use("clam")

        # Configure colors
        style.configure("TFrame", background="#1a1a2e")
        style.configure(
            "TLabel", background="#1a1a2e", foreground="#e0e0e0", font=("Segoe UI", 10)
        )
        style.configure(
            "Title.TLabel", font=("Segoe UI", 16, "bold"), foreground="#00d4ff"
        )
        style.configure("TButton", font=("Segoe UI", 10), padding=10)
        style.configure("Success.TLabel", foreground="#00ff88")
        style.configure("Error.TLabel", foreground="#ff4444")

    def create_widgets(self):
        """Create all GUI widgets."""
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title = ttk.Label(
            main_frame, text="🔑 Qwen Token Extractor", style="Title.TLabel"
        )
        title.pack(pady=(0, 20))

        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="Paste your curl command from browser DevTools below:",
            wraplength=600,
        )
        instructions.pack(anchor="w")

        # Curl input area
        self.curl_text = scrolledtext.ScrolledText(
            main_frame,
            height=8,
            font=("Consolas", 9),
            bg="#16213e",
            fg="#e0e0e0",
            insertbackground="#00d4ff",
            wrap=tk.WORD,
        )
        self.curl_text.pack(fill=tk.X, pady=10)

        # Extract button
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)

        extract_btn = tk.Button(
            btn_frame,
            text="📥 Extract & Add Account",
            command=self.extract_token,
            bg="#00d4ff",
            fg="#1a1a2e",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            padx=20,
            pady=10,
            cursor="hand2",
        )
        extract_btn.pack(side=tk.LEFT, padx=(0, 10))

        clear_btn = tk.Button(
            btn_frame,
            text="🗑️ Clear",
            command=lambda: self.curl_text.delete(1.0, tk.END),
            bg="#333",
            fg="#fff",
            font=("Segoe UI", 10),
            relief="flat",
            padx=15,
            pady=10,
            cursor="hand2",
        )
        clear_btn.pack(side=tk.LEFT)

        # Status label
        self.status_label = ttk.Label(main_frame, text="", style="Success.TLabel")
        self.status_label.pack(anchor="w", pady=5)

        # Separator
        ttk.Separator(main_frame, orient="horizontal").pack(fill=tk.X, pady=15)

        # Accounts section
        accounts_title = ttk.Label(
            main_frame, text="📋 Configured Accounts", style="Title.TLabel"
        )
        accounts_title.pack(anchor="w")

        # Accounts listbox
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        self.accounts_list = tk.Listbox(
            list_frame,
            font=("Consolas", 10),
            bg="#16213e",
            fg="#e0e0e0",
            selectbackground="#00d4ff",
            selectforeground="#1a1a2e",
            height=8,
        )
        self.accounts_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(
            list_frame, orient="vertical", command=self.accounts_list.yview
        )
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.accounts_list.config(yscrollcommand=scrollbar.set)

        # Account buttons
        acc_btn_frame = ttk.Frame(main_frame)
        acc_btn_frame.pack(fill=tk.X)

        delete_btn = tk.Button(
            acc_btn_frame,
            text="🗑️ Delete Selected",
            command=self.delete_account,
            bg="#ff4444",
            fg="#fff",
            font=("Segoe UI", 10),
            relief="flat",
            padx=15,
            pady=8,
            cursor="hand2",
        )
        delete_btn.pack(side=tk.LEFT, padx=(0, 10))

        refresh_btn = tk.Button(
            acc_btn_frame,
            text="🔄 Refresh",
            command=self.load_accounts,
            bg="#333",
            fg="#fff",
            font=("Segoe UI", 10),
            relief="flat",
            padx=15,
            pady=8,
            cursor="hand2",
        )
        refresh_btn.pack(side=tk.LEFT)

        # Account count
        self.count_label = ttk.Label(acc_btn_frame, text="")
        self.count_label.pack(side=tk.RIGHT)

    def parse_curl_cookies(self, curl_text: str) -> dict:
        """Parse cookies from a curl command string."""
        cookies = {}

        # Find the -b flag content (cookies)
        cookie_match = re.search(r'-b\s+["\^]*([^"]+)["\^]*', curl_text)
        if cookie_match:
            cookie_str = cookie_match.group(1)
            # Parse cookie pairs
            for pair in cookie_str.split(";"):
                pair = pair.strip()
                if "=" in pair:
                    key, value = pair.split("=", 1)
                    cookies[key.strip()] = value.strip()

        return cookies

    def extract_token(self):
        """Extract token from curl input."""
        curl_text = self.curl_text.get(1.0, tk.END).strip()

        if not curl_text:
            self.show_status("❌ Please paste a curl command", error=True)
            return

        cookies = self.parse_curl_cookies(curl_text)

        if not cookies.get("token"):
            self.show_status(
                "❌ Could not find 'token' cookie in curl command", error=True
            )
            return

        # Create account object with all 5 cookie fields
        account = {
            "name": f"Account ({cookies.get('aui', cookies.get('cnaui', 'unknown'))[:8]}...)",
            "cookie_token": cookies.get("token", ""),
            "cookie_cnaui": cookies.get("cnaui", ""),
            "cookie_aui": cookies.get("aui", ""),
            "cookie_bx_ua": cookies.get("bx_ua", ""),
            "cookie_bx_umidtoken": cookies.get("bx_umidtoken", ""),
        }

        # Save to config
        self.save_account(account)

        # Clear input
        self.curl_text.delete(1.0, tk.END)

        # Reload accounts list
        self.load_accounts()

    def save_account(self, account: dict):
        """Save account to config.json."""
        # Load existing config
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {
                "api_key": "sk-demo",
                "accounts": [],
                "host": "0.0.0.0",
                "port": 8001,
                "timeout": 120,
                "verbose": False,
            }

        # Initialize accounts array if not exists
        if "accounts" not in config:
            config["accounts"] = []

        # Check if account already exists (by aui)
        existing_idx = None
        for i, acc in enumerate(config["accounts"]):
            if acc.get("cookie_aui") == account["cookie_aui"]:
                existing_idx = i
                break

        if existing_idx is not None:
            config["accounts"][existing_idx] = account
            self.show_status(f"✅ Updated existing account: {account['name']}")
        else:
            config["accounts"].append(account)
            self.show_status(f"✅ Added new account: {account['name']}")

        # Save config
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    def load_accounts(self):
        """Load accounts from config.json."""
        self.accounts_list.delete(0, tk.END)

        if not self.config_path.exists():
            self.count_label.config(text="No config.json found")
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            accounts = config.get("accounts", [])

            for i, acc in enumerate(accounts):
                name = acc.get("name", f"Account {i+1}")
                aui = acc.get("cookie_aui", "")[:8]
                token_preview = acc.get("cookie_token", "")[:30] + "..."
                self.accounts_list.insert(
                    tk.END, f"[{i+1}] {name} | AUI: {aui}... | Token: {token_preview}"
                )

            self.count_label.config(text=f"Total: {len(accounts)} account(s)")

        except Exception as e:
            self.show_status(f"❌ Error loading config: {e}", error=True)

    def delete_account(self):
        """Delete selected account."""
        selection = self.accounts_list.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select an account to delete")
            return

        idx = selection[0]

        if not messagebox.askyesno("Confirm", f"Delete account #{idx + 1}?"):
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            if "accounts" in config and idx < len(config["accounts"]):
                deleted = config["accounts"].pop(idx)

                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)

                self.show_status(
                    f"✅ Deleted account: {deleted.get('name', 'Unknown')}"
                )
                self.load_accounts()

        except Exception as e:
            self.show_status(f"❌ Error deleting account: {e}", error=True)

    def show_status(self, message: str, error: bool = False):
        """Show status message."""
        self.status_label.config(
            text=message, style="Error.TLabel" if error else "Success.TLabel"
        )
        # Auto-clear after 5 seconds
        self.root.after(5000, lambda: self.status_label.config(text=""))


def main():
    root = tk.Tk()
    app = TokenExtractorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
