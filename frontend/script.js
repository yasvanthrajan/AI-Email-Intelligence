async function analyzeEmail() {
    const subjectInput = document.getElementById("subjectInput");
    const emailInput = document.getElementById("emailInput");
    const analyzeButton = document.getElementById("analyzeButton");
    const loading = document.getElementById("loading");
    const error = document.getElementById("error");
    const results = document.getElementById("results");
    const genreResults = document.getElementById("genreResults");
    const emotionResults = document.getElementById("emotionResults");

    // Get input values
    const subject = subjectInput.value.trim();
    const body = emailInput.value.trim();

    // Validate inputs
    if (!subject && !body) {
        error.textContent = "Please enter both a subject and body content before analyzing.";
        error.classList.remove("hidden");
        results.classList.add("hidden");
        return;
    }

    // Combine subject and body cleanly for backend processing
    const combinedEmailText = `Subject: ${subject}\n\n${body}`;

    // Reset UI
    error.classList.add("hidden");
    results.classList.add("hidden");
    loading.classList.remove("hidden");
    analyzeButton.disabled = true;

    try {
        // Send request to Flask backend
        const response = await fetch(
                "/predict",
            {
                method: "POST",
                headers: {
                    "Content-Type": "application/json"
                },
                body: JSON.stringify({
                    email: combinedEmailText
                })
            }
        );

        // Convert response to JSON
        const data = await response.json();

        // Handle backend error
        if (!response.ok) {
            throw new Error(
                data.error ||
                "Prediction failed."
            );
        }

        // Clear previous results
        genreResults.innerHTML = "";
        emotionResults.innerHTML = "";

        // Display genre
        if (
            data.genre &&
            data.genre.length > 0
        ) {
            data.genre.forEach(
                function(item) {
                    const div =
                        document.createElement("div");

                    div.className =
                        "result-item";

                    div.innerHTML =
                        `<span class="code">
                            ${item.code}
                         </span>
                         ${item.name}`;

                    genreResults.appendChild(div);
                }
            );
        } else {
            genreResults.innerHTML =
                "<p style='color: var(--text-muted); font-size: 14px;'>No genre detected.</p>";
        }

        // Display emotional tone
        if (
            data.emotional_tone &&
            data.emotional_tone.length > 0
        ) {
            data.emotional_tone.forEach(
                function(item) {
                    const div =
                        document.createElement("div");

                    div.className =
                        "result-item";

                    div.innerHTML =
                        `<span class="code">
                            ${item.code}
                         </span>
                         ${item.name}`;

                    emotionResults.appendChild(div);
                }
            );
        } else {
            emotionResults.innerHTML =
                "<p style='color: var(--text-muted); font-size: 14px;'>No emotional tone detected.</p>";
        }

        // Show results
        results.classList.remove("hidden");

    } catch (err) {
        console.error(err);
        error.textContent =
            "Unable to connect to the AI backend. " +
            "Make sure Flask is running.";

        error.classList.remove("hidden");
    } finally {
        loading.classList.add("hidden");
        analyzeButton.disabled = false;
    }
}