
import sys
import math

class MockScheduler:
    def __init__(self, schedule_name, patience, cooldown, factor, shape_influence, aspect_ratio=1.0):
        self.lr_schedule = schedule_name
        self.lr_patience = patience
        self.lr_cooldown = cooldown
        self.lr_factor = factor
        self.lr_shape_influence = shape_influence
        self.lr_min = 1e-8
        
        # State
        self.plateau_counter = 0
        self.cooldown_counter = 0
        self.curr_lr = 1e-3
        
        # Logic from LearnedRoundingConverter
        if shape_influence > 0:
            ar_factor = math.sqrt(aspect_ratio)
            blend = shape_influence
            self.effective_patience = self.lr_patience
            raw_factor = self.lr_factor
            aggressive_factor = raw_factor ** ar_factor
            self.effective_factor = raw_factor + (aggressive_factor - raw_factor) * blend
            self.effective_cooldown = self.lr_cooldown
        else:
            self.effective_patience = self.lr_patience
            self.effective_factor = self.lr_factor
            self.effective_cooldown = self.lr_cooldown
        
        print(f"Schedule: {self.lr_schedule} | Patience: {self.effective_patience} | Factor: {self.effective_factor:.4f}")

    def step(self, improved):
        if improved:
            self.plateau_counter = 0
            return "Improved"
        
        self.plateau_counter += 1
        
        if self.lr_schedule == "plateau":
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                return "Cooldown"
            elif self.plateau_counter >= self.effective_patience:
                old_lr = self.curr_lr
                self.curr_lr = max(self.curr_lr * self.effective_factor, self.lr_min)
                self.cooldown_counter = self.effective_cooldown
                self.plateau_counter = 0
                return f"DECAY ({old_lr:.2e} -> {self.curr_lr:.2e})"
            return f"Wait ({self.plateau_counter}/{self.effective_patience})"
        elif self.lr_schedule == "exponential":
            old_lr = self.curr_lr
            self.curr_lr = max(self.curr_lr * 0.99, self.lr_min) # Mock gamma 0.99
            return f"EXP DECAY ({old_lr:.2e} -> {self.curr_lr:.2e})"
        else:
            return "Adaptive (Assume Update)"

def test_repro():
    print("\n--- Test 1: User Case (Patience=3) ---")
    s1 = MockScheduler("plateau", patience=3, cooldown=0, factor=0.96, shape_influence=10.5)
    for i in range(1, 5):
        print(f"S1.{i}: {s1.step(improved=False)}")

    print("\n--- Test 2: Patience=0 (Hypothesis: defaulted to 0) ---")
    s2 = MockScheduler("plateau", patience=0, cooldown=0, factor=0.96, shape_influence=10.5)
    for i in range(1, 4):
        print(f"S2.{i}: {s2.step(improved=False)}")

    print("\n--- Test 3: Wrong Schedule (Defaults to Exponential?) ---")
    s3 = MockScheduler("exponential", patience=3, cooldown=0, factor=0.96, shape_influence=10.5)
    for i in range(1, 4):
        print(f"S3.{i}: {s3.step(improved=False)}")

if __name__ == "__main__":
    test_repro()
