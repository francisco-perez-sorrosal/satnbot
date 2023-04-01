import pickle


class ChatHistory:
    def __init__(self, history_file):
        try:
            print(f"Loading command history from {history_file}")
            with open(history_file, "rb") as file:
                self.history = pickle.load(file)
                print(self.history)
        except FileNotFoundError:
            print("No history!")
            self.history = {}

    def save_history(self):
        with open(f"history.pkl", "wb") as file:
            pickle.dump(self.history, file)

    def get_session(self, session_name):
        return self.history.get(session_name, [])

    def add_to_session(self, session_name, c):
        """Add an entry to the history for a given session."""
        if session_name not in self.history:
            self.history[session_name] = []
        self.history[session_name].append(c)

    def get_history(self, session_name):
        """Return the history for a given session."""
        return self.history[session_name]
