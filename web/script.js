fetch("../output/results.json")
.then(res => res.json())
.then(data => {

    let risky = data.filter(d => d.anomaly === -1);

    document.getElementById("stats").innerHTML =
        `Total Events: ${data.length} <br> Risky Events: ${risky.length}`;

    const ctx = document.getElementById('chart');

    new Chart(ctx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Flight Risk',
                data: data.map(d => ({
                    x: d.distance,
                    y: d.alt_diff
                }))
            }]
        }
    });
});