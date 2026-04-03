fetch("results.json")
  .then(response => response.json())
  .then(data => {
    console.log("DATA:", data);

    const acc = document.getElementById("accuracy");
    const samp = document.getElementById("samples");
    const dur = document.getElementById("duration");

    if (acc) acc.innerText = data.model_accuracy;
    if (samp) samp.innerText = data.samples_used;
    if (dur) dur.innerText = data.avg_duration + " minutes";
  })
  .catch(error => {
    console.error("Error loading JSON:", error);
  });