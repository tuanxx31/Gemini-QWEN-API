"""
Seedream 4.5 Token Extractor - GUI Version
Extract and manage Seedream (BytePlus Ark) account tokens from curl commands.
"""

import re
import json
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from pathlib import Path


class TokenExtractorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🌠 Seedream Token Extractor")
        self.root.geometry("800x700")
        self.root.configure(bg="#0f172a")

        self.config_path = Path(__file__).parent / "config.json"

        self.setup_styles()
        self.create_widgets()
        self.load_accounts()

    def setup_styles(self):
        """Setup custom styles."""
        style = ttk.Style()
        style.theme_use("clam")

        # Configure colors
        style.configure("TFrame", background="#0f172a")
        style.configure(
            "TLabel", background="#0f172a", foreground="#94a3b8", font=("Segoe UI", 10)
        )
        style.configure(
            "Title.TLabel", font=("Segoe UI", 16, "bold"), foreground="#6366f1"
        )
        style.configure("TButton", font=("Segoe UI", 10), padding=10)
        style.configure("Success.TLabel", foreground="#10b981", background="#0f172a")
        style.configure("Error.TLabel", foreground="#ef4444", background="#0f172a")

    def create_widgets(self):
        """Create all GUI widgets."""
        main_frame = ttk.Frame(self.root, padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title = ttk.Label(
            main_frame, text="🌠 Seedream 4.5 Token Extractor", style="Title.TLabel"
        )
        title.pack(pady=(0, 20))

        # Instructions
        instructions = ttk.Label(
            main_frame,
            text="Paste your curl command from BytePlus Console DevTools (Network tab) below:",
            wraplength=750,
        )
        instructions.pack(anchor="w")

        # Curl input area
        self.curl_text = scrolledtext.ScrolledText(
            main_frame,
            height=12,
            font=("Consolas", 9),
            bg="#1e293b",
            fg="#f8fafc",
            insertbackground="#6366f1",
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
            bg="#6366f1",
            fg="white",
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
            bg="#334155",
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
            bg="#1e293b",
            fg="#f8fafc",
            selectbackground="#6366f1",
            selectforeground="white",
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
            bg="#ef4444",
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
            bg="#334155",
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

    def extract_token(self):
        """Extract cookies and tokens from curl input."""
        curl_text = self.curl_text.get(1.0, tk.END).strip()

        if not curl_text:
            self.show_status("❌ Please paste a curl command", error=True)
            return

        # Extract Cookies
        cookie_match = re.search(r'-b\s+["\^]*([^"\s\^]+[^"]*)["\^]*' , curl_text)
        if not cookie_match:
             cookie_match = re.search(r'-H\s+["\^]*cookie:\s*([^"\s\^]+[^"]*)["\^]*', curl_text, re.IGNORECASE)
        
        cookie = cookie_match.group(1).strip() if cookie_match else ""
        if not cookie:
            self.show_status("❌ Could not find Cookies", error=True)
            return

        # Extract CSRF Token
        csrf_match = re.search(r'x-csrf-token:\s*([^"\s\^]+)', curl_text, re.IGNORECASE)
        if not csrf_match:
            csrf_match = re.search(r'x-csrf-token:\s*"([^"]+)"', curl_text, re.IGNORECASE)
        csrf_token = csrf_match.group(1).strip() if csrf_match else ""

        # Extract Web ID
        web_id_match = re.search(r'x-web-id:\s*([^"\s\^]+)', curl_text, re.IGNORECASE)
        if not web_id_match:
            web_id_match = re.search(r'x-web-id:\s*"([^"]+)"', curl_text, re.IGNORECASE)
        web_id = web_id_match.group(1).strip() if web_id_match else ""

        # Extract Endpoint ID and Model from data-raw
        endpoint_id = "seedream-4-5-251128"
        model = "seedream-4-5"
        
        data_match = re.search(r'--data-raw\s+["\^]*(.*?)["\^]*$', curl_text, re.DOTALL)
        if data_match:
            data_str = data_match.group(1).replace("^", "")
            try:
                data_json = json.loads(data_str)
                endpoint_id = data_json.get("EndpointId", endpoint_id)
                model = data_json.get("Model", model)
            except:
                pass

        account = {
            "name": f"Seedream ({cookie[:15]}...)",
            "cookie": cookie,
            "csrf_token": csrf_token,
            "web_id": web_id,
            "endpoint_id": endpoint_id,
            "model": model
        }

        # Save to config
        self.save_account(account)

        # Clear input
        self.curl_text.delete(1.0, tk.END)

        # Reload accounts list
        self.load_accounts()

    def save_account(self, account: dict):
        """Save account to config.json."""
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        else:
            config = {
                "host": "0.0.0.0",
                "port": 8002,
                "api_key": "sk-seedream-secret",
                "verbose": True,
                "timeout": 120,
                "accounts": []
            }

        if "accounts" not in config:
            config["accounts"] = []

        # Check for duplicate by cookie
        existing_idx = None
        for i, acc in enumerate(config["accounts"]):
            if acc.get("cookie") == account["cookie"]:
                existing_idx = i
                break

        if existing_idx is not None:
            config["accounts"][existing_idx] = account
            self.show_status(f"✅ Updated existing account")
        else:
            config["accounts"].append(account)
            self.show_status(f"✅ Added new account")

        # Save config
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4, ensure_ascii=False)

    def load_accounts(self):
        """Load accounts from config.json."""
        self.accounts_list.delete(0, tk.END)

        if not self.config_path.exists():
            self.count_label.config(text="No config found")
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            accounts = config.get("accounts", [])

            for i, acc in enumerate(accounts):
                name = acc.get("name", f"Account {i+1}")
                self.accounts_list.insert(
                    tk.END, f"[{i+1}] {name} | Model: {acc.get('model')}"
                )

            self.count_label.config(text=f"Total: {len(accounts)} account(s)")

        except Exception as e:
            self.show_status(f"❌ Error loading config: {e}", error=True)

    def delete_account(self):
        """Delete selected account."""
        selection = self.accounts_list.curselection()
        if not selection:
            return

        idx = selection[0]
        if not messagebox.askyesno("Confirm", "Delete this account?"):
            return

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            if "accounts" in config and idx < len(config["accounts"]):
                config["accounts"].pop(idx)
                with open(self.config_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, indent=4, ensure_ascii=False)
                self.show_status("✅ Deleted account")
                self.load_accounts()
        except Exception as e:
            self.show_status(f"❌ Error: {e}", error=True)

    def show_status(self, message: str, error: bool = False):
        """Show status message."""
        self.status_label.config(
            text=message, style="Error.TLabel" if error else "Success.TLabel"
        )
        self.root.after(5000, lambda: self.status_label.config(text=""))


def main():
    root = tk.Tk()
    app = TokenExtractorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
