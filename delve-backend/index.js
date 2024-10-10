const express = require("express");
const { PythonShell } = require("python-shell");
const path = require("path");
const app = express();
const PORT = 5000;

// Endpoint to handle search queries and return links
app.get("/query", (req, res) => {
  const query = req.query.query;

  // Log received query
  console.log("Received query:", query);

  // PythonShell options to run the script with the query as an argument
  let options = {
    mode: "text",
    pythonPath: "/Users/habibi/opt/anaconda3/bin/python3", // Adjust this to your correct Python path if needed
    scriptPath: path.join(__dirname),
    args: [query], // Pass the query to the Python script
  };

  // Run the Python script
  PythonShell.run("query.py", options, function (err, results) {
    if (err) {
      console.error("Error running Python script:", err);
      return res.status(500).json({ error: "Error executing Python script" });
    }

    // Log the raw output from the Python script
    console.log("Python script raw results:", results);

    // Parse the results and return the links to the frontend
    try {
      if (results && results.length > 0) {
        const links = JSON.parse(results[0]); // Assuming the first line of output is the JSON string
        console.log("Parsed links from Python script:", links);

        // Respond with the parsed links
        return res.json({ links });
      } else {
        console.error("Python script returned no results.");
        return res.status(500).json({ error: "No results from Python script" });
      }
    } catch (parseError) {
      console.error("Error parsing Python script output:", parseError);
      return res.status(500).json({ error: "Error parsing Python script output" });
    }
  });
});

// Start the Express server
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
