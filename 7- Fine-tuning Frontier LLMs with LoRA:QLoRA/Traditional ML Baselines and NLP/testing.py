# Tester: runs any item→number function on 250 test items,
# reports errors (abs & log), colors results, and plots preds vs truth.
#
# Usage:
#   def predict(item):
#       return my_estimate
#
#   Tester.test(predict)




import math
import matplotlib.pyplot as plt


# Constants
COLOR_MAP = {
    "green": "\033[92m",
    "orange": "\033[93m",
    "red": "\033[91m",
}
RESET = "\033[0m"
SIZE = 250




class Tester:

    def __init__(self, predictor, data, title=None, size=SIZE):
        """Initialize storage and settings for evaluating a predictor."""
        self.predictor = predictor
        self.data = data
        self.title = title or predictor.__name__.replace("_", " ").title()
        self.size = size
        self.predictions = []
        self.truths = []
        self.absolute_errors = []
        self.squared_log_errors = []
        self.colors = []


    def get_color_for_error(self, error, truth):
        """Return 'green', 'orange', or 'red' based on error thresholds."""
        if error < 40 or (error / truth) < 0.2:
            return "green"
        elif error < 80 or (error / truth) < 0.4:
            return "orange"
        else:
            return "red"


    def run_data_example(self, i):
        """Compute prediction, truth, errors, color, and log them for a single record."""
        data_example = self.data[i]
        pred = self.predictor(data_example)
        truth = data_example.price
        error = abs(pred - truth)
        log_error = math.log(truth + 1) - math.log(pred + 1)
        sq_log_error = log_error ** 2
        color = self.get_color_for_error(error, truth)

        # shorten long titles
        title = data_example.title if len(data_example.title) <= 40 else data_example.title[:40]+"..."

        # Store
        self.predictions.append(pred)
        self.truths.append(truth)
        self.absolute_errors.append(error)
        self.squared_log_errors.append(sq_log_error)
        self.colors.append(color)
        print(f"{COLOR_MAP[color]}"
              f"{i+1}: Pred: ${pred:,.2f}"
              f"Truth=${truth:,.2f} "
              f"Error=${error:,.2f} "
              f"SLE={sq_log_error:.2f} "
              f"Item={title}{RESET}")


    def chart(self, title):
        """Show a scatter + diagonal plot of truth vs. guesses, colored by accuracy."""
        #max_error = max(self.absolute_errors)
        plt.figure(figsize=(12, 8))
        max_val = max(max(self.truths), max(self.predictions))
        plt.plot([0, max_val], [0, max_val], color="deepskyblue", lw=2, alpha=0.6)
        plt.scatter(self.truths, self.predictions, s=3, c=self.colors)
        plt.xlabel("Ground Truth")
        plt.ylabel("Model Estimate")
        plt.xlim(0, max_val)
        plt.ylim(0, max_val)
        plt.title(title)
        plt.show()


    def summarize_result(self):
        """Compute overall metrics (avg error, RMSLE, hit rate) and plot them."""
        avg_error = sum(self.absolute_errors) / self.size
        # Root Mean Squared Logarithmic Error
        rmsle = math.sqrt(sum(self.squared_log_errors) / self.size)
        correct_count = sum(1 for c in self.colors if c == "green")
        accuracy = correct_count / self.size * 100

        summary_title = (
            f"{self.title}  "
            f"Error=${avg_error:,.2f}  "
            f"RMSLE={rmsle:.2f}  "
            f"Accuracy={accuracy:.1f}%"
        )
        self.chart(summary_title)


    def run(self):
        """Evaluate all dataExamples in sequence, then summarize."""
        for i in range(self.size):
            self.run_data_example(i)
        self.summarize_result()

    @classmethod
    def test(cls, function, data):
        """Quick entry point: instantiate with default data/size and run."""
        cls(function, data).run()




