// Data definitions
const stateCapacity = [
    { state: 'Tamil Nadu', capacity: '10,000+ MW' },
    { state: 'Gujarat', capacity: '10,000+ MW' },
    { state: 'Karnataka', capacity: '5,300 MW' },
    { state: 'Maharashtra', capacity: '5,000 MW' },
    { state: 'Rajasthan', capacity: '4,700 MW' }
];

const windFarms = [
    { id: 'muppandal', name: 'Muppandal Wind Farm', state: 'Tamil Nadu', district: 'Kanyakumari', lat: 8.2435, lng: 77.5458, capacity: 1500, owner: 'Multiple (TNEB, Suzlon, Private)', category: 'multi' },
    { id: 'jaisalmer', name: 'Jaisalmer Wind Park', state: 'Rajasthan', district: 'Jaisalmer', lat: 26.9157, lng: 70.9083, capacity: 1064, owner: 'Suzlon Energy, Adani Green', category: 'oem' },
    { id: 'brahmanvel', name: 'Brahmanvel Wind Farm', state: 'Maharashtra', district: 'Dhule', lat: 20.9450, lng: 74.0750, capacity: 528, owner: 'Parakh Agro, Suzlon', category: 'oem' },
    { id: 'dhalgaon', name: 'Dhalgaon Wind Farm', state: 'Maharashtra', district: 'Sangli', lat: 17.1512, lng: 74.7431, capacity: 278, owner: 'Gadre Marine, Suzlon', category: 'oem' },
    { id: 'vankusawade', name: 'Vankusawade Wind Park', state: 'Maharashtra', district: 'Satara', lat: 17.4160, lng: 73.8120, capacity: 259, owner: 'Suzlon Energy', category: 'oem' },
    { id: 'kayathar', name: 'Kayathar Wind Farm', state: 'Tamil Nadu', district: 'Thoothukudi', lat: 8.9500, lng: 77.7200, capacity: 300, owner: 'Tata Power, Renew Power', category: 'ipp' },
    { id: 'mamatkheda', name: 'Mamatkheda Wind Park', state: 'Madhya Pradesh', district: 'Ratlam', lat: 23.5800, lng: 74.9500, capacity: 100, owner: 'Orange Renewables', category: 'ipp' },
    { id: 'beluguppa', name: 'Beluguppa Wind Park', state: 'Andhra Pradesh', district: 'Anantapur', lat: 14.7100, lng: 77.1300, capacity: 100, owner: 'Tata Power', category: 'ipp' },
    { id: 'vasadra', name: 'Vasadra Wind Farm', state: 'Gujarat', district: 'Kutch', lat: 23.1000, lng: 69.8000, capacity: 150, owner: 'Adani Green Energy', category: 'ipp' },
    { id: 'tirunelveli', name: 'Tirunelveli Clusters', state: 'Tamil Nadu', district: 'Tirunelveli', lat: 8.7139, lng: 77.7567, capacity: 1200, owner: 'Multiple (Inox Wind, ReNew)', category: 'multi' },
    { id: 'dangri', name: 'Dangri Wind Farm', state: 'Rajasthan', district: 'Jaisalmer', lat: 26.5000, lng: 71.0000, capacity: 54, owner: 'Oil India Ltd', category: 'psu' },
    { id: 'bercha', name: 'Bercha Wind Farm', state: 'Madhya Pradesh', district: 'Ratlam', lat: 23.4700, lng: 75.1400, capacity: 50, owner: 'Inox Wind', category: 'oem' },
    { id: 'khavda', name: 'Khavda Mega Park (Hybrid)', state: 'Gujarat', district: 'Kutch', lat: 23.8300, lng: 69.5300, capacity: 30000, owner: 'Multiple (Mega Park)', category: 'mega' }
];

const categoryColors = {
    'psu': '#0ea5e9',    // Cyan
    'ipp': '#10b981',    // Emerald
    'oem': '#8b5cf6',    // Purple
    'multi': '#f59e0b',  // Amber
    'mega': '#d946ef'    // Fuchsia
};

let activeFarm = null;

// Initialize UI
function initUI() {
    const listEl = document.getElementById('capacity-list');
    stateCapacity.forEach(item => {
        const li = document.createElement('li');
        li.innerHTML = `<span>${item.state}</span> <span class="capacity-val">${item.capacity}</span>`;
        listEl.appendChild(li);
    });

    document.getElementById('close-detail').addEventListener('click', () => {
        document.getElementById('detail-overlay').classList.add('hidden');
    });
    
    document.getElementById('close-results').addEventListener('click', () => {
        document.getElementById('results-dashboard').classList.add('hidden');
    });

    // Toggle custom upload section based on radio selection
    const radioButtons = document.querySelectorAll('input[name="model_mode"]');
    const uploadZone = document.getElementById('upload-zone');
    
    radioButtons.forEach(radio => {
        radio.addEventListener('change', (e) => {
            if(e.target.value === 'custom') {
                uploadZone.classList.remove('hidden');
            } else {
                uploadZone.classList.add('hidden');
            }
        });
    });

    // Execute Pipeline Button Logic
    document.getElementById('run-pipeline-btn').addEventListener('click', executePipeline);
}

async function executePipeline() {
    if(!activeFarm) return;
    
    const isCustom = document.querySelector('input[name="model_mode"]:checked').value === 'custom';
    const loadingState = document.getElementById('loading-state');
    const loadingText = document.getElementById('loading-text');
    const progressFill = document.querySelector('.progress-fill');
    
    loadingState.classList.remove('hidden');
    progressFill.style.width = '10%';
    
    let url = '';
    let options = {};
    
    if(isCustom) {
        const fileInput = document.getElementById('scada-file');
        if(!fileInput.files.length) {
            alert('Please upload a CSV file with your SCADA actuals to retrain the model.');
            loadingState.classList.add('hidden');
            return;
        }
        loadingText.innerText = "Uploading data...";
        
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);
        
        url = `http://localhost:8000/api/v1/retrain/${activeFarm.id}`;
        options = { method: 'POST', body: formData };
    } else {
        loadingText.innerText = "Queueing operational forecast job...";
        url = `http://localhost:8000/api/v1/forecast/${activeFarm.id}`;
        options = { method: 'POST' };
    }
    
    try {
        const response = await fetch(url, options);
        const data = await response.json();
        
        if (data.job_id) {
            pollJobStatus(data.job_id);
        } else {
            alert("Pipeline error: " + (data.error || "Unknown error"));
            loadingState.classList.add('hidden');
        }
    } catch (err) {
        alert("Failed to connect to FastAPI Server. Is it running on port 8000?");
        loadingState.classList.add('hidden');
    }
}

async function pollJobStatus(jobId) {
    const loadingText = document.getElementById('loading-text');
    const progressFill = document.querySelector('.progress-fill');
    const loadingState = document.getElementById('loading-state');

    const interval = setInterval(async () => {
        try {
            const resp = await fetch(`http://localhost:8000/api/v1/job-status/${jobId}`);
            const data = await resp.json();

            if (data.status === 'completed') {
                clearInterval(interval);
                progressFill.style.width = '100%';
                loadingText.innerText = "Finalizing dashboard...";
                
                setTimeout(() => {
                    loadingState.classList.add('hidden');
                    progressFill.style.width = '0%';
                    document.getElementById('detail-overlay').classList.add('hidden');
                    showResultsDashboard(data.result);
                }, 1000);
            } else if (data.status === 'failed') {
                clearInterval(interval);
                alert("Pipeline Failed: " + (data.error || "Unknown error"));
                loadingState.classList.add('hidden');
            } else {
                // Update message and increment progress visually
                loadingText.innerText = data.message || "Processing...";
                const currentWidth = parseInt(progressFill.style.width);
                if (currentWidth < 90) {
                    progressFill.style.width = (currentWidth + 5) + '%';
                }
            }
        } catch (err) {
            console.error("Polling error:", err);
        }
    }, 2000);
}

function showResultsDashboard(data) {
    document.getElementById('results-dashboard').classList.remove('hidden');
    document.getElementById('pipeline-badge').classList.remove('hidden');
    document.getElementById('results-title').innerText = `Operational 96-Block Forecast: ${activeFarm.name}`;
    
    // Dynamically load the generated charts from the local outputs folder
    // Note: To prevent caching issues if generating repeatedly, append timestamp
    const ts = new Date().getTime();
    document.getElementById('img-financials').src = `../outputs/plots/${activeFarm.id}/financial_savings_comparison.png?t=${ts}`;
    document.getElementById('img-timeseries').src = `../outputs/plots/${activeFarm.id}/forecast_vs_actual_timeseries.png?t=${ts}`;
    document.getElementById('img-live-forecast').src = `../outputs/plots/${activeFarm.id}/live_96_block_forecast.png?t=${ts}`;
    
    // Update KPI values using the exact numbers calculated by the Python Backend!
    document.getElementById('val-physics').innerText = `₹ ${data.physics_penalty.toLocaleString('en-IN', {maximumFractionDigits: 0})}`;
    document.getElementById('val-ml').innerText = `₹ ${data.ml_penalty.toLocaleString('en-IN', {maximumFractionDigits: 0})}`;
    document.getElementById('val-savings').innerText = `₹ ${data.savings.toLocaleString('en-IN', {maximumFractionDigits: 0})}`;
    
    // Set the Download Button Link!
    document.getElementById('download-csv-btn').href = `http://localhost:8000/api/v1/download-forecast/${activeFarm.id}`;
}

// Initialize Map
function initMap() {
    // Centered on India
    const map = L.map('map', {
        zoomControl: false // Move zoom control if needed, or hide for cleaner UI
    }).setView([22.5937, 78.9629], 5);

    L.control.zoom({
        position: 'bottomright'
    }).addTo(map);

    // Dark Matter tile layer for premium dark mode aesthetics
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(map);

    // Add markers as buffer zones (Circles)
    windFarms.forEach(farm => {
        const color = categoryColors[farm.category] || '#ffffff';
        
        // Radius scale based on capacity, minimum 10km (10000m) buffer for visibility
        // Base buffer size + log scaled based on MW
        const radiusInMeters = 10000 + (Math.log(farm.capacity) * 2000);

        const circle = L.circle([farm.lat, farm.lng], {
            color: color,
            fillColor: color,
            fillOpacity: 0.2,
            weight: 2,
            radius: radiusInMeters,
            className: 'animated-circle' // Custom CSS animation
        }).addTo(map);

        // Tooltip on hover
        circle.bindTooltip(`<strong>${farm.name}</strong><br>${farm.capacity} MW`, {
            className: 'custom-tooltip',
            direction: 'top'
        });

        // Click event to show details
        circle.on('click', () => {
            showDetails(farm, color);
            // Optional flyTo for dynamic interaction
            map.flyTo([farm.lat, farm.lng], 8, { duration: 1.5 });
        });
    });
}

// Show detailed info in overlay
function showDetails(farm, color) {
    activeFarm = farm;
    const overlay = document.getElementById('detail-overlay');
    const content = document.getElementById('detail-content');
    const commandCenter = document.getElementById('command-center');

    content.innerHTML = `
        <h3 style="color: ${color}">${farm.name}</h3>
        <p>${farm.district}, ${farm.state}</p>
        <div class="detail-metric">${farm.capacity.toLocaleString()} MW</div>
        <p><strong>Ownership:</strong><br>${farm.owner}</p>
        <p><strong>Coordinates:</strong><br>${farm.lat.toFixed(4)}° N, ${farm.lng.toFixed(4)}° E</p>
    `;

    overlay.classList.remove('hidden');
    commandCenter.classList.remove('hidden');
}

// Run on load
document.addEventListener('DOMContentLoaded', () => {
    initUI();
    initMap();
});
