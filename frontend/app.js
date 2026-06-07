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
    { id: 'khavda', name: 'Khavda Mega Park (Hybrid)', state: 'Gujarat', district: 'Kutch', lat: 23.8300, lng: 69.5300, capacity: 30000, owner: 'Multiple (Mega Park)', category: 'mega' },
    { id: 'rsopl_koppal', name: 'RSOPL Koppal Wind Plant', state: 'Karnataka', district: 'Koppal', lat: 15.3400, lng: 76.1500, capacity: 75, owner: 'ReNew Surya Ojas Pvt. Ltd.', category: 'ipp' }
];

const categoryColors = {
    'psu': '#0ea5e9',    // Cyan
    'ipp': '#10b981',    // Emerald
    'oem': '#8b5cf6',    // Purple
    'multi': '#f59e0b',  // Amber
    'mega': '#d946ef'    // Fuchsia
};

let activeFarm = null;
window.mapInstance = null;

// Initialize UI
function initUI() {
    // Populate state capacity section
    const listEl = document.getElementById('capacity-list');
    stateCapacity.forEach(item => {
        const li = document.createElement('li');
        li.innerHTML = `<span>${item.state}</span> <span class="capacity-val">${item.capacity}</span>`;
        listEl.appendChild(li);
    });

    // Populate wind farm explorer sidebar list
    const farmsListEl = document.getElementById('farms-list');
    windFarms.forEach(farm => {
        const li = document.createElement('li');
        li.dataset.id = farm.id;
        li.dataset.state = farm.state;
        li.innerHTML = `
            <span class="farm-name">${farm.name}</span>
            <span class="farm-capacity">${farm.capacity.toLocaleString()} MW</span>
        `;
        li.addEventListener('click', () => {
            selectWindFarm(farm);
        });
        farmsListEl.appendChild(li);
    });

    // Setup detail overlay closing
    document.getElementById('close-detail').addEventListener('click', () => {
        document.getElementById('detail-overlay').classList.add('hidden');
        document.getElementById('command-center').classList.add('hidden');
        
        // Remove active styling in sidebar explorer
        document.querySelectorAll('#farms-list li').forEach(li => {
            li.classList.remove('active');
        });
        activeFarm = null;
    });
    
    // Setup results modal closing
    document.getElementById('close-results').addEventListener('click', () => {
        document.getElementById('results-dashboard').classList.add('hidden');
        document.getElementById('modal-backdrop').classList.add('hidden');
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

    // Initialize search filter
    initSearch();

    // Check API Connection immediately and every 5 seconds
    checkApiConnection();
    setInterval(checkApiConnection, 5000);

    // Setup Wind Flow toggle click handler
    document.getElementById('wind-flow-toggle').addEventListener('click', toggleWindFlow);

    // Setup Windy filter dropdown listener
    const filterSelect = document.getElementById('windy-filter-select');
    if (filterSelect) {
        filterSelect.addEventListener('change', (e) => {
            const iframe = document.getElementById('windy-iframe');
            if (iframe) {
                iframe.className = `theme-${e.target.value}`;
            }
        });
        // Initial setup
        const iframe = document.getElementById('windy-iframe');
        if (iframe) {
            iframe.className = `theme-${filterSelect.value}`;
        }
    }

    // Pre-fetch wind vectors for the details card display
    fetch('http://localhost:8000/api/v1/wind-vectors')
        .then(r => r.json())
        .then(data => {
            window.windVectorsCache = data;
        })
        .catch(err => console.error("Silent wind vectors pre-fetch failed:", err));
}

// Check Backend FastAPI connection health
async function checkApiConnection() {
    const statusEl = document.getElementById('api-status');
    const dot = statusEl.querySelector('.status-dot');
    const text = statusEl.querySelector('.status-text');
    
    try {
        // Query the job-status with a dummy health check key.
        // It responds with 200 (error message) if the FastAPI server is online.
        const response = await fetch('http://localhost:8000/api/v1/job-status/health-check');
        if (response.ok || response.status === 200) {
            dot.className = 'status-dot online';
            text.innerText = 'API Connected';
        } else {
            dot.className = 'status-dot offline';
            text.innerText = 'API Server Error';
        }
    } catch (err) {
        dot.className = 'status-dot offline';
        text.innerText = 'API Offline';
    }
}

// Search filter logic for the sidebar list
function initSearch() {
    const searchInput = document.getElementById('farm-search');
    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase().trim();
        const items = document.querySelectorAll('#farms-list li');
        
        items.forEach(item => {
            const name = item.querySelector('.farm-name').innerText.toLowerCase();
            const state = item.dataset.state.toLowerCase();
            if (name.includes(query) || state.includes(query)) {
                item.style.display = 'flex';
            } else {
                item.style.display = 'none';
            }
        });
    });
}

// Unified function to select a wind farm from sidebar or map
function selectWindFarm(farm) {
    activeFarm = farm;
    
    // Add active highlight in sidebar list
    document.querySelectorAll('#farms-list li').forEach(li => {
        if (li.dataset.id === farm.id) {
            li.classList.add('active');
            li.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            li.classList.remove('active');
        }
    });
    
    const color = categoryColors[farm.category] || '#ffffff';
    showDetails(farm, color);
    
    // Check if custom model exists and show/hide the radio option
    const customOption = document.getElementById('custom-model-option');
    if (customOption) {
        fetch(`http://localhost:8000/api/v1/custom-model-exists/${farm.id}`)
            .then(r => r.json())
            .then(data => {
                if (data.exists) {
                    customOption.classList.remove('hidden');
                } else {
                    customOption.classList.add('hidden');
                    // Reset selection to pretrained if custom option is currently selected
                    const selectedMode = document.querySelector('input[name="model_mode"]:checked');
                    if (selectedMode && selectedMode.value === 'custom_model') {
                        document.querySelector('input[value="pretrained"]').checked = true;
                    }
                }
            })
            .catch(err => {
                console.error("Failed to check if custom model exists:", err);
                customOption.classList.add('hidden');
            });
    }
    
    // Smoothly fly map to location
    if (window.mapInstance) {
        window.mapInstance.flyTo([farm.lat, farm.lng], 8, { duration: 1.5 });
    }
    
    // Show marker tooltip
    if (farm.marker) {
        farm.marker.openTooltip();
    }
}

async function executePipeline() {
    if(!activeFarm) return;
    
    const modelMode = document.querySelector('input[name="model_mode"]:checked').value;
    const isCustomUpload = modelMode === 'custom';
    const loadingState = document.getElementById('loading-state');
    const loadingText = document.getElementById('loading-text');
    const progressFill = document.querySelector('.progress-fill');
    
    loadingState.classList.remove('hidden');
    progressFill.style.width = '10%';
    
    let url = '';
    let options = {};
    
    if(isCustomUpload) {
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
        const isCustomModel = modelMode === 'custom_model';
        loadingText.innerText = `Queueing operational forecast job (${isCustomModel ? 'custom model' : 'pretrained model'})...`;
        url = `http://localhost:8000/api/v1/forecast/${activeFarm.id}?model_mode=${isCustomModel ? 'custom' : 'pretrained'}`;
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
    document.getElementById('modal-backdrop').classList.remove('hidden');
    document.getElementById('pipeline-badge').classList.remove('hidden');
    document.getElementById('results-title').innerText = `Operational Report: ${activeFarm.name}`;
    
    // Determine suffix based on selected model mode
    const modelMode = document.querySelector('input[name="model_mode"]:checked').value;
    const modelSuffix = (modelMode === 'custom_model' || modelMode === 'custom') ? 'custom' : 'pretrained';
    
    // Dynamically load the generated charts from the local outputs folder
    const ts = new Date().getTime();
    document.getElementById('img-financials').src = `../outputs/plots/${activeFarm.id}/financial_savings_comparison_${modelSuffix}.png?t=${ts}`;
    document.getElementById('img-timeseries').src = `../outputs/plots/${activeFarm.id}/forecast_vs_actual_timeseries_${modelSuffix}.png?t=${ts}`;
    document.getElementById('img-live-forecast').src = `../outputs/plots/${activeFarm.id}/live_96_block_forecast_${modelSuffix}.png?t=${ts}`;
    
    // Update KPI values using numbers calculated by the backend
    document.getElementById('val-physics').innerText = `₹ ${data.physics_penalty.toLocaleString('en-IN', {maximumFractionDigits: 0})}`;
    document.getElementById('val-ml').innerText = `₹ ${data.ml_penalty.toLocaleString('en-IN', {maximumFractionDigits: 0})}`;
    document.getElementById('val-savings').innerText = `₹ ${data.savings.toLocaleString('en-IN', {maximumFractionDigits: 0})}`;
    
    // Calculate and format tomorrow's date (for the Day-Ahead Operational Forecast)
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);
    const dateOptions = { day: 'numeric', month: 'short', year: 'numeric' };
    document.getElementById('val-live-date').innerText = tomorrow.toLocaleDateString('en-IN', dateOptions);
    
    // Set live stats returned from GFS live weather model fetch
    if (data.live_stats) {
        document.getElementById('val-live-peak').innerText = `${data.live_stats.max_mw.toFixed(1)} MW`;
        document.getElementById('val-live-avg').innerText = `${data.live_stats.avg_mw.toFixed(1)} MW`;
    } else {
        document.getElementById('val-live-peak').innerText = `-- MW`;
        document.getElementById('val-live-avg').innerText = `-- MW`;
    }
    
    // Initialize default tab state to "Tomorrow's Live Forecast (Operational)"
    switchTab(null, 'tab-live');
    document.querySelectorAll('.tab-btn').forEach(btn => {
        if (btn.innerText.includes('Tomorrow')) {
            btn.classList.add('active');
        } else {
            btn.classList.remove('active');
        }
    });

    // Set the Download Button Link
    document.getElementById('download-csv-btn').href = `http://localhost:8000/api/v1/download-forecast/${activeFarm.id}?model_mode=${modelSuffix}`;
}

// Global Tab switcher function
function switchTab(event, tabId) {
    // Hide all tab contents
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    // Deactivate all tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show current tab content
    document.getElementById(tabId).classList.add('active');
    
    // Activate clicked button
    if (event) {
        event.currentTarget.classList.add('active');
    }
}
window.switchTab = switchTab;

// Initialize Map
function initMap() {
    // Centered on India
    const map = L.map('map', {
        zoomControl: false
    }).setView([22.5937, 78.9629], 5);

    window.mapInstance = map;

    L.control.zoom({
        position: 'bottomright'
    }).addTo(map);

    window.baseTileLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 19
    }).addTo(map);

    // Sync with Windy iframe when map moves/zooms
    map.on('moveend', () => {
        const btn = document.getElementById('wind-flow-toggle');
        if (btn && btn.classList.contains('active')) {
            updateWindyIframe();
        }
    });

    // Add markers as buffer zones (Circles)
    windFarms.forEach(farm => {
        const color = categoryColors[farm.category] || '#ffffff';
        
        // Radius scale based on capacity, minimum 10km (10000m) buffer for visibility
        const radiusInMeters = 10000 + (Math.log(farm.capacity) * 2000);

        const circle = L.circle([farm.lat, farm.lng], {
            color: color,
            fillColor: color,
            fillOpacity: 0.2,
            weight: 2,
            radius: radiusInMeters,
            className: 'animated-circle'
        }).addTo(map);

        // Store circle reference to allow map activation from sidebar explorer
        farm.marker = circle;

        // Tooltip on hover
        circle.bindTooltip(`<strong>${farm.name}</strong><br>${farm.capacity} MW`, {
            className: 'custom-tooltip',
            direction: 'top'
        });

        // Click event to show details
        circle.on('click', () => {
            selectWindFarm(farm);
        });
    });
}

// Show detailed info in overlay
function showDetails(farm, color) {
    activeFarm = farm;
    const overlay = document.getElementById('detail-overlay');
    const content = document.getElementById('detail-content');
    const commandCenter = document.getElementById('command-center');

    // Dynamic premium glow matching the category color
    overlay.style.borderColor = `${color}66`;
    overlay.style.boxShadow = `0 8px 32px rgba(0, 0, 0, 0.5), 0 0 15px ${color}33`;

    let windText = "";
    if (window.windVectorsCache) {
        const windData = window.windVectorsCache.find(w => w.id === farm.id);
        if (windData) {
            const compassDir = getCompassDirection(windData.wind_direction);
            windText = `
                <div class="glass-divider" style="margin: 0.75rem 0;"></div>
                <p style="margin-bottom: 0.40rem; font-weight: 500; font-size: 0.85rem; color: var(--accent-glow);">Live Meteorological Forecast</p>
                <div style="display: flex; justify-content: space-between; align-items: center; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 8px;">
                    <div>
                        <span class="micro-text" style="display:block;margin-bottom:2px;font-size:0.7rem;">Wind Speed</span>
                        <span style="font-weight:600;font-size:0.85rem;color:var(--text-main);">${windData.wind_speed.toFixed(1)} m/s</span>
                    </div>
                    <div style="text-align: right;">
                        <span class="micro-text" style="display:block;margin-bottom:2px;font-size:0.7rem;">Direction</span>
                        <span style="font-weight:600;font-size:0.85rem;color:var(--text-main);">${windData.wind_direction.toFixed(0)}° (${compassDir})</span>
                    </div>
                </div>
            `;
        }
    }

    content.innerHTML = `
        <h3 style="color: ${color}">${farm.name}</h3>
        <p style="margin-bottom: 0.5rem; color: var(--text-muted);">${farm.district}, ${farm.state}</p>
        <div class="detail-metric" style="background: ${color}1a; border-color: ${color}33; color: ${color};">${farm.capacity.toLocaleString()} MW</div>
        <p style="margin-top: 0.75rem;"><strong>Ownership:</strong><br>${farm.owner}</p>
        <p style="margin-bottom: 0.50rem;"><strong>Coordinates:</strong><br>${farm.lat.toFixed(4)}° N, ${farm.lng.toFixed(4)}° E</p>
        ${windText}
    `;

    overlay.classList.remove('hidden');
    commandCenter.classList.remove('hidden');
}

// Wind flow overlay global variables
window.windVectorsCache = null;

// Sync with Windy iframe when map moves/zooms
function updateWindyIframe() {
    if (!window.mapInstance) return;
    const windyIframe = document.getElementById('windy-iframe');
    const center = window.mapInstance.getCenter();
    const zoom = window.mapInstance.getZoom();
    
    const newSrc = `https://embed.windy.com/embed2.html?lat=${center.lat.toFixed(4)}&lon=${center.lng.toFixed(4)}&zoom=${zoom}&level=surface&overlay=wind&menu=&message=true&marker=&calendar=&pressure=&type=map&location=coordinates&detail=&detailLat=&detailLon=&metricWind=default&metricTemp=default&radarRange=`;
    
    if (windyIframe.src !== newSrc) {
        windyIframe.src = newSrc;
    }
}

// Toggle live Windy.com radar map overlay
async function toggleWindFlow() {
    const btn = document.getElementById('wind-flow-toggle');
    const windyContainer = document.getElementById('windy-container');
    const mapEl = document.getElementById('map');
    const themeSelector = document.getElementById('windy-theme-selector');
    const isActive = btn.classList.contains('active');
    
    if (isActive) {
        btn.classList.remove('active');
        btn.innerHTML = `<svg viewBox="0 0 24 24" width="12" height="12" style="margin-right: 2px;"><path fill="currentColor" d="M19.36 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.64-4.96zM19 18H6c-2.21 0-4-1.79-4-4 0-2.05 1.53-3.76 3.56-3.97l1.07-.11.5-.95C8.08 7.14 9.94 6 12 6c2.62 0 4.88 1.86 5.39 4.43l.3 1.5 1.53.11c1.56.1 2.78 1.41 2.78 2.96 0 1.65-1.35 3-3 3z"/></svg> Show Wind Flow`;
        
        windyContainer.classList.add('hidden');
        if (themeSelector) themeSelector.classList.add('hidden');
        mapEl.classList.remove('transparent-bg');
        if (window.baseTileLayer) {
            window.baseTileLayer.setOpacity(1);
        }
    } else {
        btn.classList.add('active');
        btn.innerHTML = `<svg viewBox="0 0 24 24" width="12" height="12" style="margin-right: 2px;"><path fill="currentColor" d="M19.36 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.64-4.96zM19 18H6c-2.21 0-4-1.79-4-4 0-2.05 1.53-3.76 3.56-3.97l1.07-.11.5-.95C8.08 7.14 9.94 6 12 6c2.62 0 4.88 1.86 5.39 4.43l.3 1.5 1.53.11c1.56.1 2.78 1.41 2.78 2.96 0 1.65-1.35 3-3 3z"/></svg> Hide Wind Flow`;
        
        // Synchronize and update Windy source
        updateWindyIframe();
        
        windyContainer.classList.remove('hidden');
        if (themeSelector) themeSelector.classList.remove('hidden');
        mapEl.classList.add('transparent-bg');
        if (window.baseTileLayer) {
            window.baseTileLayer.setOpacity(0.15); // Keep base layer at 0.15 opacity for premium overlay blending
        }
    }
}

// Resolve degrees to standard 16-point meteorological compass sectors
function getCompassDirection(deg) {
    const sectors = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"];
    const idx = Math.round(deg / 22.5) % 16;
    return sectors[idx];
}

// Run on load
document.addEventListener('DOMContentLoaded', () => {
    initUI();
    initMap();
});
