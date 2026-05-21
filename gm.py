from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS 
import ee 
import datetime
import math
import numpy as np
from functools import lru_cache
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM,Dense
    # ================= GEE OPTIMIZATION V5 =================

def maskS2clouds(img):

        scl=img.select('SCL')

        mask=scl.neq(9)\
                .And(scl.neq(10))

        return img.updateMask(mask)\
                .divide(10000)\
                .copyProperties(
                        img,
                        ["system:time_start"]
                )
    # ================= CẤU HÌNH GOOGLE EARTH ENGINE =================
PROJECT_ID = 'baitap-470705'

try:
        ee.Initialize(project=PROJECT_ID)
        print("✅ GEE CONNECTED")
except Exception as e:
        print("🔄 Đang xác thực...")
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)

app = Flask(__name__)
CORS(app)

    # ================= GIAO DIỆN FORESTGUARD PRO V3.5 =================
HTML = """
    <!DOCTYPE html>
    <html lang="vi">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>ForestGuard Pro v3.5 - Dashboard Phân Tích</title>
        <link
            rel="stylesheet"
            href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        />
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
        <script
            src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">
        </script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            :root { 
                --dark-green: #0b251c; 
                --forest: #1b4332; 
                --leaf: #2d6a4f; 
                --mint: #95d5b2; 
                --accent: #d8f3dc; 
                --danger: #e74c3c; 
                --co2-blue: #3498db; 
                --gold: #f1c40f;
                --panel-bg: rgba(11, 37, 28, 0.98);
            }
            
            body { margin:0; display:flex; height:100vh; background:#000; font-family: 'Plus Jakarta Sans', sans-serif; color: white; overflow:hidden; }
            
            /* Sidebar Styles */
          #sidebar{
    width:320px;
    min-width:320px;
    padding:20px;
    background:#071b15;
    overflow-y:auto;
    border-right:1px solid rgba(255,255,255,.05);
    backdrop-filter:blur(20px);
    box-shadow:5px 0 30px rgba(0,0,0,.4);
    z-index:1000;
}

            ;

            z-index:1000;

            }
           #map{
    flex:1;
    height:100vh;
    position:relative;
    overflow:hidden;
    z-index:1;
}

            .leaflet-container{
                background:#111;
            }

            .leaflet-container{
                width:100%;
                height:100%;
            }

            #analysis-panel{
                width:450px;
                min-width:450px;
                overflow-y:auto;
                display:none;
                z-index:999;
            }
            
            /* Analysis Panel - Modern Dashboard Style */
            #analysis-panel { 
                width:450px; 
                background: var(--panel-bg); 
                padding:24px; 
                overflow-y:auto; 
                border-left: 1px solid rgba(255,255,255,0.1); 
                display:none;
                box-shadow: -10px 0 30px rgba(0,0,0,0.5);
            }
            
            .brand { font-weight: 800; color: var(--mint); font-size: 1.4rem; margin-bottom: 25px; letter-spacing: -1px; display: flex; align-items: center; }
            .brand span { font-size: 0.6rem; margin-left: 8px; border: 1px solid var(--mint); padding: 2px 6px; border-radius: 4px; }
            
            .section { background: rgba(255,255,255,0.03); padding:15px; border-radius:14px; margin-bottom:15px; border: 1px solid rgba(255,255,255,0.05); }
            .label-custom { color: var(--mint); font-weight: 700; font-size: 0.7rem; text-transform: uppercase; margin-bottom: 8px; display: block; opacity: 0.8; letter-spacing: 0.5px; }
            
            .form-select { background: #123327; border: 1px solid #2d6a4f; color: white; font-size: 0.9rem; border-radius: 8px; padding: 10px; }
            #province-container { max-height: 150px; overflow-y: auto; background: rgba(0,0,0,0.2); padding: 10px; border-radius: 8px; margin-top: 8px; }
            
            .btn-run { 
    background: linear-gradient(135deg,#40916c,#1b4332);
    color:white;
    border:none;
    width:100%;
    height:55px;
    border-radius:12px;

    font-weight:700;
    font-size:15px;

    display:flex;
    justify-content:center;
    align-items:center;

    text-align:center;
    padding:0 15px;

    margin-top:25px;

    white-space:normal;
    line-height:1.2;

    transition:all .3s ease;
    box-shadow:0 4px 15px rgba(0,0,0,.2);
}
            .btn-run:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(64, 145, 108, 0.4); filter: brightness(1.1); }

            /* Dashboard Cards */
            .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-bottom: 20px; }
            .stat-card-new { 
                background: rgba(255,255,255,0.05); 
                padding: 18px; 
                border-radius: 16px; 
                border: 1px solid rgba(149, 213, 178, 0.1);
                position: relative;
                overflow: hidden;
            }
            .stat-card-new::before {
                content: '';
                position: absolute;
                left: 0; top: 20%; height: 60%; width: 4px;
                background: var(--mint);
                border-radius: 0 4px 4px 0;
            }
            .stat-card-new.wide { grid-column: span 2; }
            .card-val { font-size: 1.8rem; font-weight: 800; color: #fff; line-height: 1.2; margin-top: 5px; }
            .card-unit { font-size: 0.8rem; color: var(--mint); font-weight: 500; margin-left: 4px; }
            
            /* AI Commentary Box */
            .ai-box { 
                background: linear-gradient(145deg, rgba(45, 106, 79, 0.2), rgba(11, 37, 28, 0.4));
                border-radius: 16px;
                padding: 20px;
                border-left: 5px solid var(--mint);
                margin-bottom: 25px;
            }
            .ai-commentary-text { font-size: 0.95rem; line-height: 1.6; color: #e0f2f1; font-weight: 500; }

            .chart-wrapper { background: rgba(0,0,0,0.2); border-radius: 12px; padding: 15px; margin-bottom: 15px; height: 220px; }
            
            #loader { display:none; position: fixed; top:0; left:0; width:100%; height:100%; background: rgba(0,0,0,0.85); z-index: 9999; justify-content: center; align-items: center; flex-direction: column; backdrop-filter: blur(5px); }
            .spinner-custom { width: 4rem; height: 4rem; border-width: 0.35em; color: var(--mint) !important; }

            /* Custom Scrollbar */
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: transparent; }
            ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 10px; }
            ::-webkit-scrollbar-thumb:hover { background: var(--mint); }
            /* ===== AI FORECAST BUTTON ===== */

#forecast-btn{

position:absolute;

left:340px;
bottom:25px;

width:58px;
height:58px;

border-radius:50%;

background:linear-gradient(
135deg,
#1b4332,
#2d6a4f
);

border:2px solid #95d5b2;

display:flex;

justify-content:center;

align-items:center;

font-size:28px;

cursor:pointer;

z-index:9999;

box-shadow:
0 0 20px rgba(
46,
204,
113,
.5
);

transition:.3s;

}

#forecast-btn:hover{

transform:scale(1.1);

box-shadow:
0 0 35px rgba(
46,
204,
113,
.9
);

}

/* ===== AI NEWS PANEL ===== */

#forecast-panel{

position:absolute;

left:420px;

bottom:20px;

width:420px;

max-height:500px;

overflow-y:auto;

display:none;

z-index:9999;

padding:20px;

border-radius:20px;

background:
rgba(
11,
37,
28,
0.97
);

backdrop-filter:
blur(20px);

border:
1px solid rgba(
149,
213,
178,
0.2
);

box-shadow:
0 0 40px rgba(
0,
0,
0,
.5
);

}

.forecast-title{

font-size:22px;

font-weight:800;

color:#95d5b2;

margin-bottom:15px;

}

.forecast-close{

float:right;

cursor:pointer;

font-size:22px;

color:#fff;

}

.forecast-item{

background:
rgba(
255,
255,
255,
0.04
);

padding:12px;

border-radius:12px;

margin-bottom:10px;

line-height:1.5;

}

/* ===== COMMERCIAL UI UPGRADE ===== */
#sidebar, #analysis-panel, #forecast-panel{
    background: linear-gradient(180deg, rgba(6,31,24,.98), rgba(3,18,14,.98));
}

.section, .stat-card-new, .ai-box, .forecast-item{
    box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 10px 30px rgba(0,0,0,.18);
}

.form-select{
    height:48px;
    font-weight:600;
    transition:.25s;
}

.form-select:focus{
    background:#123327;
    color:#fff;
    border-color:#95d5b2;
    box-shadow:0 0 0 .2rem rgba(149,213,178,.14);
}

.province-row{
    display:flex;
    align-items:center;
    gap:10px;
    padding:10px 12px;
    margin-bottom:8px;
    border-radius:12px;
    background:rgba(255,255,255,.035);
    border:1px solid rgba(149,213,178,.08);
    transition:.25s ease;
}

.province-row:hover{
    background:rgba(149,213,178,.10);
    border-color:rgba(149,213,178,.28);
    transform:translateX(2px);
}

.province-check{
    width:18px;
    height:18px;
    cursor:pointer;
    border:1px solid rgba(149,213,178,.6);
}

.province-label{
    flex:1;
    display:flex;
    justify-content:space-between;
    align-items:center;
    cursor:pointer;
    font-size:.88rem;
    color:#fff;
    margin:0;
}

.province-label small{
    color:#95d5b2;
    font-size:.65rem;
    opacity:.8;
}

.btn-run{
    background:linear-gradient(135deg,#52b788,#2d6a4f,#1b4332);
    letter-spacing:.2px;
}

#forecast-panel{
    width:480px;
    max-height:620px;
}

.forecast-model-card{
    border-left:4px solid #95d5b2;
}

.forecast-data-card{
    border-left:4px solid #52b788;
}

.leaflet-interactive{
    filter: drop-shadow(0 0 6px rgba(149,213,178,.55));
}

        </style>
    </head>
    <body>

    <div id="loader">
        <div class="spinner-border spinner-custom mb-4"></div>
        <div id="loader-text" class="fw-bold h5 text-white">ĐANG TẢI DỮ LIỆU ĐA PHỔ...</div>
        <div class="text-white-50 small">Vui lòng đợi trong giây lát</div>
    </div>

    <div id="sidebar">
        <div class="brand">🌲 ForestGuard <span>PRO V3.5</span></div>
        
        <div class="section">
            <label class="label-custom">Khu vực quan trắc</label>
            <select id="country" class="form-select mb-2" onchange="fetchProvinces()">
                <option value="">Chọn quốc gia...</option>
            </select>
            <div id="province-container"></div>
        </div>

        <div class="section">

        <div class="row g-2">

            <!-- Năm -->
            <div class="col-6">

                <label class="label-custom">
                    Năm
                </label>

                <select
                    id="year"
                    class="form-select"
                    
                >
                    <option value="2026">
                        2026
                    </option>
                    <option value="2025">
                        2025
                    </option>

                    <option
                        value="2024"
                    selected
                    >
                        2024
                    </option>

                    <option value="2023">
                        2023
                    </option>

                    <option value="2022">
                        2022
                    </option>

                </select>

            </div>

            <!-- Tháng -->
            <div class="col-6">

                <label class="label-custom">
                    Tháng
                </label>

                <select
                    id="month"
                    class="form-select"
                    
                >

                    <option value="1">Tháng 1</option>
                    <option value="2">Tháng 2</option>
                    <option value="3">Tháng 3</option>
                    <option value="4">Tháng 4</option>
                    <option value="5">Tháng 5</option>
                    <option value="6">Tháng 6</option>
                    <option value="7">Tháng 7</option>
                    <option value="8">Tháng 8</option>
                    <option value="9">Tháng 9</option>
                    <option value="10">Tháng 10</option>
                    <option value="11">Tháng 11</option>
                    <option value="12">Tháng 12</option>

               </select>

               </div>

        </div>

    </div>
            <label class="label-custom">Chỉ số hiển thị Layer</label>
            <select id="index-type" class="form-select">
                <option value="NDVI">Thảm thực vật (NDVI)</option>
                <option value="BIOMASS">Sinh khối (Tấn/Ha)</option>
                <option value="CO2">Hấp thụ CO2 (Tấn/Ha)</option>
            </select>
        

        <button class="btn-run" onclick="processAnalysis()">
🚀 Chạy phân tích
</button>
    </div>

    <div id="map"></div>
    <div id="forecast-btn">

🔮

</div>

<div id="forecast-panel">

<div class="forecast-title">

📰 BẢN TIN AI DỰ BÁO

<span
class="forecast-close"
onclick="closeForecast()"
>

✖

</span>

</div>

<div id="forecast-content">

<div class="forecast-item">

AI đang chờ phân tích...

</div>

</div>

</div>
    <div id="analysis-panel">
        <div class="d-flex justify-content-between align-items-center mb-4">
            <h4 class="fw-800 m-0" style="color: var(--mint); letter-spacing: -0.5px;">BÁO CÁO PHÂN TÍCH</h4>
            <span class="badge bg-success" style="border-radius: 6px; padding: 6px 10px;">LIVE DATA</span>
        </div>

        <div class="dashboard-grid">
            <div class="stat-card-new wide">
                <label class="label-custom">Tổng diện tích rừng quản lý</label>
                <div class="d-flex align-items-baseline">
                    <span class="card-val" id="forest-ha">0</span>
                    <span class="card-unit">Hecta (Ha)</span>
                </div>
            </div>
            <div class="stat-card-new">
                <label class="label-custom" id="metric-label">NDVI TB</label>
                <div class="d-flex align-items-baseline">
                    <span class="card-val" id="metric-val">0</span>
                    <span class="card-unit" id="metric-unit">-</span>
                </div>
            </div>
            <div class="stat-card-new">
                <label class="label-custom">Sức khỏe rừng</label>
                <div class="card-val" id="health-status" style="font-size: 1.2rem; color: var(--mint);">---</div>
            </div>
        </div>

        <div class="ai-box">
            <label class="label-custom" style="color: white; opacity: 1; margin-bottom: 12px;">🤖 NHẬN XÉT HỆ THỐNG</label>
            <div id="ai-commentary" class="ai-commentary-text">Khởi tạo dữ liệu để bắt đầu phân tích...</div>
        </div>

        <label class="label-custom">Diễn biến Sinh khối (Tấn/Ha)</label>
        <div class="chart-wrapper"><canvas id="biomassChart"></canvas></div>

        <label class="label-custom">Tương quan NDVI & CO2</label>
        <div class="chart-wrapper"><canvas id="co2NdviChart"></canvas></div>

    <script>
var currentLayer=null;
var provinceBoundaryLayer=null;
var charts={};
        var map=L.map(
        'map',
        {
        zoomControl:false,
        attributionControl:false
        }
        ).setView([15.5,108],6);
        const baseLayer = L.tileLayer(
            'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            {
                maxZoom:19,
                attribution:'&copy; OpenStreetMap contributors'
            }
        ).addTo(map);

        baseLayer.on(
            "tileerror",
            function(e){
                console.error(
                    "Lỗi tải bản đồ nền:",
                    e
                );
            }
        );
        window.addEventListener(

        'resize',

        ()=>{

        map.invalidateSize(true);

        }

        );
    

        // Fetch initial countries
        fetch('/api/countries').then(r => r.json()).then(data => {
            const select = document.getElementById("country");
            data.forEach(c => select.add(new Option(c, c)));
        });

        function fetchProvinces() {
            const country = document.getElementById("country").value;
            const container = document.getElementById("province-container");

            if(provinceBoundaryLayer){
                map.removeLayer(provinceBoundaryLayer);
                provinceBoundaryLayer = null;
            }

            container.innerHTML = "<div class='text-white-50 small p-2'>Đang tải danh sách tỉnh...</div>";

            fetch(`/api/provinces?country=${encodeURIComponent(country)}`)
            .then(async r => {
    if(!r.ok){
        const text = await r.text();
        console.error("API analyze lỗi:", text);
        throw new Error("API analyze lỗi " + r.status);
    }
    return r.json();
})
            .then(data => {
                container.innerHTML = "";

                data.provinces.forEach(p => {
                    const safeId = "p-" + p.replace(/\s+/g, "-").replace(/[^a-zA-Z0-9-_]/g, "");

                    container.innerHTML += `
                    <div class='province-row'>
                        <input
                            class='form-check-input province-check'
                            type='checkbox'
                            name='province'
                            value='${p}'
                            id='${safeId}'
                            onchange="zoomToProvince(this)"
                        >

                        <label
                            class='form-check-label province-label'
                            for='${safeId}'
                        >
                            <span>${p}</span>
                            <small>Zoom ranh giới</small>
                        </label>
                    </div>`;
                });
            })
            .catch(err => {
                console.log(err);
                container.innerHTML = "<div class='text-danger small p-2'>Không tải được danh sách tỉnh</div>";
            });
        }

        function zoomToProvince(inputEl){
            const provinceName = inputEl.value;
            const country = document.getElementById("country").value;

            if(!inputEl.checked){
                if(provinceBoundaryLayer){
                    map.removeLayer(provinceBoundaryLayer);
                    provinceBoundaryLayer = null;
                }
                return;
            }

            document.querySelectorAll('input[name="province"]').forEach(cb => {
                if(cb !== inputEl) cb.checked = false;
            });

            fetch(`/api/province-boundary?country=${encodeURIComponent(country)}&province=${encodeURIComponent(provinceName)}`)
            .then(r => r.json())
            .then(data => {

                if(provinceBoundaryLayer){
                    map.removeLayer(provinceBoundaryLayer);
                }

                provinceBoundaryLayer = L.geoJSON(data.geojson, {
                    style: {
                        color: "#95d5b2",
                        weight: 3,
                        opacity: 1,
                        fillColor: "#2d6a4f",
                        fillOpacity: 0.12,
                        dashArray: "8,5"
                    }
                }).addTo(map);

                map.fitBounds([
                    [data.bounds[1], data.bounds[0]],
                    [data.bounds[3], data.bounds[2]]
                ], {
                    padding: [40, 40],
                    animate: true
                });

            })
            .catch(err => {
                console.log(err);
                alert("Không tải được ranh giới tỉnh");
            });
        }

        function updateChart(id, type, labels, datasets, extraOptions = {}) {
            if(charts[id]) charts[id].destroy();
            const ctx = document.getElementById(id).getContext('2d');
            
            const defaultOptions = { 
                responsive: true, 
                maintainAspectRatio: false,
                plugins: { 
                    legend: { 
                        display: datasets.length > 1, 
                        labels: { color: '#ccc', font: { family: 'Plus Jakarta Sans', size: 10 }, usePointStyle: true } 
                    } 
                },
                scales: { 
                    y: { 
                        grid: { color: 'rgba(255,255,255,0.05)', drawBorder: false }, 
                        ticks: { color: '#888', font: { size: 9 } } 
                    },
                    x: { 
                        grid: { display: false }, 
                        ticks: { color: '#888', font: { size: 9 } } 
                    }
                }
            };

            charts[id] = new Chart(ctx, {
                type: type,
                data: { labels: labels, datasets: datasets },
                options: Object.assign(defaultOptions, extraOptions)
            });
        }
        function processAnalysis(){

    const payload={

        country:
        document.getElementById(
            "country"
        ).value,

        provinces:
        [...document.querySelectorAll(
            'input[name="province"]:checked'
        )].map(
            e=>e.value
        ),

        year:
        document.getElementById(
            "year"
        ).value,

        month:
        document.getElementById(
            "month"
        ).value,

        type:
        document.getElementById(
            "index-type"
        ).value
    };

    if(
        !payload.country ||
        payload.provinces.length===0
    ){

        alert(
            "Vui lòng chọn khu vực"
        );

        return;
    }

    document.getElementById(
        "loader"
    ).style.display="flex";

    fetch(

        "/api/analyze",

        {

            method:"POST",

            headers:{

                "Content-Type":
                "application/json"

            },

            body:
            JSON.stringify(
                payload
            )
        }

    )

    .then(
        r=>r.json()
    )

    .then(data=>{

        document.getElementById(
            "loader"
        ).style.display="none";

        document.getElementById(
            "analysis-panel"
        ).style.display="block";

        if(currentLayer){

            map.removeLayer(
                currentLayer
            );

        }

if(!data.map){
    console.error("Không có URL layer GEE:", data);
    alert("Không lấy được layer ảnh từ GEE");
    return;
}

currentLayer = L.tileLayer(data.map, {
    opacity: 1,
    maxZoom: 20,
    errorTileUrl: ""
});

currentLayer.on("tileerror", function(e){
    console.error("Lỗi tile GEE:", e);
});

currentLayer.addTo(map);

        map.fitBounds([

            [data.bounds[1],data.bounds[0]],

            [data.bounds[3],data.bounds[2]]

        ]);

        document.getElementById(
            "forest-ha"
        ).innerText=
        data.stats.forest_ha.toLocaleString();

        document.getElementById(
            "metric-val"
        ).innerText=
        data.stats.current_val;

        document.getElementById(
            "metric-unit"
        ).innerText=
        data.stats.unit;

        document.getElementById(
            "metric-label"
        ).innerText=
        payload.type+" TRUNG BÌNH";

        document.getElementById(
            "ai-commentary"
        ).innerHTML=
        data.comment;

        let health="TỐT";

        if(
            data.stats.health_idx<0.4
        )
        health="KÉM";

        else if(
            data.stats.health_idx<0.6
        )
        health="TRUNG BÌNH";

        document.getElementById(
            "health-status"
        ).innerText=
        health;
        
        let forecastHtml = "";

        if(data.decision_news){
            forecastHtml += data.decision_news;
        }

        if(data.forecast){

            forecastHtml += `
            <div class='forecast-item forecast-model-card'>
                <b>🔮 Dữ liệu AI dự báo 6 tháng tiếp theo</b><br>
                <span>Mô hình: GEE + RandomForest + LSTM</span>
            </div>
            `;

            data.forecast.forEach(f=>{

                forecastHtml += `
                <div class='forecast-item forecast-data-card'>
                    <b>📅 ${f.month}/${f.year}</b><br>
                    🌿 NDVI dự báo: <b>${f.ndvi}</b><br>
                    📦 Biomass dự báo: <b>${f.biomass} Tấn/Ha</b><br>
                    ☁️ CO2 dự báo: <b>${f.co2} Tấn/Ha</b>
                </div>
                `;

            });
        }

        document
        .getElementById("forecast-content")
        .innerHTML = forecastHtml || "<div class='forecast-item'>Chưa có dữ liệu dự báo.</div>";
        updateChart(

            'biomassChart',

            'bar',

            data.time_series.months,

            [{

                label:'Sinh khối',

                data:
                data.time_series.biomass,

                backgroundColor:
                '#2d6a4f'

            }]
        );

        updateChart(

            'co2NdviChart',

            'line',

            data.time_series.months,

            [

            {

                label:'NDVI',

                data:
                data.time_series.ndvi,

                borderColor:
                '#2ecc71'

            },

            {

                label:'CO2',

                data:
                data.time_series.co2,

                borderColor:
                '#3498db'

            }

            ]

        );

    })

    .catch(err=>{

        console.log(err);

        document.getElementById(
            "loader"
        ).style.display="none";

        alert(
            "Lỗi phân tích dữ liệu"
        );

    });

}
        

                
        // Pixel inspector on click
        map.on('click', function(e) {
            const lat = e.latlng.lat;
            const lng = e.latlng.lng;
            const year = document.getElementById("year").value;
            const popup = L.popup().setLatLng(e.latlng).setContent("<div class='text-dark'>Đang trích xuất...</div>").openOn(map);

            fetch('/api/pixel-history', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({ lat, lng, year })
            })
            .then(r => r.json())
            .then(data => {
                const content = `
                    <div style="min-width:180px; color:#fff; background: #0b251c; padding: 12px; border-radius: 10px;">
                        <div style="font-size:0.7rem; color:#95d5b2; font-weight:700; text-transform:uppercase">Tọa độ điểm</div>
                        <div style="font-size:0.9rem; margin-bottom:8px">${lat.toFixed(4)}, ${lng.toFixed(4)}</div>
                        <hr style="margin:8px 0; border-color:rgba(255,255,255,0.1)">
                        <div class="d-flex justify-content-between"><span>🌿 NDVI:</span> <b>${data.avg_ndvi}</b></div>
                        <div class="d-flex justify-content-between"><span>📦 Biomass:</span> <b>${data.avg_biomass}</b></div>
                        <div class="d-flex justify-content-between"><span>☁️ CO2:</span> <b>${data.avg_co2}</b></div>
                    </div>`;
                popup.setContent(content);
            });
        });
        // mở popup

document
.getElementById(
"forecast-btn"
)

.onclick=function(){

const panel=

document.getElementById(
"forecast-panel"
);

panel.style.display="block";

};

// đóng popup

function closeForecast(){

document
.getElementById(
"forecast-panel"
)
.style.display="none";

}
    </script>
    </body>
    </html>
    """

    # ================= BACKEND LOGIC (HƠN 150 DÒNG TIẾP THEO) =================

def interpolate_data(data_list):
        """Xử lý dữ liệu trống trong chuỗi thời gian vệ tinh"""
        arr = np.array(data_list, dtype=float)
        for i in range(len(arr)):
            if arr[i] <= 0 or np.isnan(arr[i]):
                if i > 0: arr[i] = arr[i-1]
                else:
                    next_vals = arr[i:][arr[i:] > 0]
                    arr[i] = next_vals[0] if len(next_vals) > 0 else 0
        return arr.round(3).tolist()
# ===== HYBRID GEOAI TRAIN DATA =====

def build_training_data(region):

    X=[]
    y=[]

    years=[2022,2023,2024,2025,2026]

    for year in years:

        max_month=12

        if year==2026:
            max_month=3

        for m in range(1,max_month+1):

            ndvi=get_satellite_data(
                region,
                year,
                m
            )

            stat=ndvi.reduceRegion(
                reducer=ee.Reducer.mean(),
                geometry=region,
                scale=1000,
                bestEffort=True
            )

            val=stat.get("NDVI")

            try:
                val=val.getInfo()
            except:
                val=0

            if val is None:
                val=0

            biomass=10.5*np.exp(
                3.2*val
            )

            co2=biomass*1.7249

            X.append([

                year,
                m,
                val,
                biomass,
                co2

            ])

            y.append([

                val,
                biomass,
                co2

            ])

    return np.array(X),np.array(y)
def train_hybrid_ai(region):

    X,y=build_training_data(
        region
    )

    X_train,\
    X_test,\
    y_train,\
    y_test = train_test_split(

        X,
        y,

        test_size=0.3,

        random_state=42,

        shuffle=False
    )

    rf=RandomForestRegressor(

        n_estimators=100,

        random_state=42
    )

    rf.fit(
        X_train,
        y_train
    )

    scaler=MinMaxScaler()

    X_scale=scaler.fit_transform(X)

    X_lstm=[]
    y_lstm=[]

    seq=4

    for i in range(len(X_scale)-seq):

        X_lstm.append(
            X_scale[i:i+seq]
        )

        y_lstm.append(
            X_scale[i+seq][2:5]
        )

    X_lstm=np.array(X_lstm)
    y_lstm=np.array(y_lstm)

    model=Sequential()

    model.add(
        Input(
            shape=(
                seq,
                X_lstm.shape[2]
            )
        )
    )

    model.add(
        LSTM(
            64,
            return_sequences=True
        )
    )

    model.add(
        LSTM(32)
    )

    model.add(
        Dense(
            32,
            activation='relu'
        )
    )

    model.add(
        Dense(3)
    )

    model.compile(
        optimizer='adam',
        loss='mse'
    )

    model.fit(
        X_lstm,
        y_lstm,
        epochs=30,
        verbose=0
    )

    return rf,model,scaler,X_scale
def predict_6months(

    region,
    year,
    month

):

    rf, model, scaler, X = train_hybrid_ai(
    region
)

    last_seq=X[-4:]

    forecast=[]

    current_month=int(
        month
    )

    current_year=int(
        year
    )

    for i in range(6):

        pred=model.predict(

            np.expand_dims(
                last_seq,
                axis=0
            ),

            verbose=0

        )[0]

        dummy=np.zeros((1,5))
        dummy[0,2:5]=pred

        pred_full=scaler.inverse_transform(dummy)[0]

        pred=[
            pred_full[2],
            pred_full[3],
            pred_full[4]
        ]

        current_month+=1

        if current_month>12:

            current_month=1

            current_year+=1

        forecast.append({

            "month":
            current_month,

            "year":
            current_year,

            "ndvi":
            round(pred[0],3),

            "biomass":
            round(pred[1],2),

            "co2":
            round(pred[2],2)

        })
        new_data=[
            current_year,
            current_month,
            pred[0],
            pred[1],
            pred[2]
        ]

        new=scaler.transform([new_data])

        last_seq=np.vstack([

            last_seq[1:],
            new

        ])

    return forecast
def generate_ai_decision_news(region_name, year, month, stats, forecast):
    ndvi = stats["health_idx"]
    forest_ha = stats["forest_ha"]
    change = stats["forest_change"]

    if ndvi >= 0.7:
        health = "TỐT"
        risk = "THẤP"
        action = "Duy trì giám sát định kỳ và ưu tiên bảo vệ vùng rừng hiện có."
    elif ndvi >= 0.5:
        health = "TRUNG BÌNH"
        risk = "TRUNG BÌNH"
        action = "Cần theo dõi các vùng NDVI thấp, kiểm tra thực địa nếu xu hướng giảm tiếp tục."
    else:
        health = "KÉM"
        risk = "CAO"
        action = "Ưu tiên kiểm tra thực địa, khoanh vùng suy giảm và lập kế hoạch phục hồi rừng."

    trend = "tăng" if change > 0 else "giảm" if change < 0 else "ổn định"

    next_ndvi = forecast[0]["ndvi"] if forecast else ndvi
    next_risk = "tăng" if next_ndvi < ndvi else "giảm hoặc ổn định"

    return f"""
    <div class='forecast-item'>
        <b>📍 Khu vực:</b> {region_name}<br>
        <b>📅 Thời gian phân tích:</b> {month}/{year}<br>
        <b>🌲 Diện tích rừng:</b> {forest_ha:,} ha
    </div>

    <div class='forecast-item'>
        <b>🌿 Đánh giá sức khỏe rừng</b><br>
        NDVI trung bình đạt <b>{ndvi}</b>, hệ thống đánh giá rừng ở mức 
        <b>{health}</b>. So với cùng kỳ năm trước, diện tích rừng có xu hướng 
        <b>{trend} {abs(change)}%</b>.
    </div>

    <div class='forecast-item'>
        <b>⚠️ Cảnh báo AI</b><br>
        Mức rủi ro hiện tại: <b>{risk}</b>.<br>
        Dự báo tháng tiếp theo cho thấy nguy cơ suy giảm có xu hướng 
        <b>{next_risk}</b>.
    </div>

    <div class='forecast-item'>
        <b>✅ Khuyến nghị quản lý</b><br>
        - {action}<br>
        - Ưu tiên giám sát vùng có NDVI thấp hoặc biến động mạnh.<br>
        - Kết hợp dữ liệu mưa, nhiệt độ, cháy rừng và kiểm tra thực địa.<br>
        - Dùng kết quả này làm cơ sở hỗ trợ ra quyết định quản lý rừng.
    </div>
    """
def maskL8(img):

    qa = img.select('QA_PIXEL')

    mask = qa.bitwiseAnd(1<<3).eq(0)\
            .And(
                qa.bitwiseAnd(1<<4).eq(0)
            )

    return img.updateMask(mask)\
            .multiply(0.0000275)\
            .add(-0.2)\
            .copyProperties(
                img,
                ["system:time_start"]
            )
def get_satellite_data(
    region,
    year,
    month,
    image_date=None
):

    year=int(year)
    month=int(month)

    start=ee.Date.fromYMD(
        year,
        month,
        1
    )

    end=start.advance(
        1,
        'month'
    )

    collection=(

    ee.ImageCollection(
        "COPERNICUS/S2_SR_HARMONIZED"
    )

    .filterBounds(region)

    .filterDate(
        start,
        end
    )

    .filter(
        ee.Filter.lt(
            'CLOUDY_PIXEL_PERCENTAGE',
            80  
        )
    )

    .map(
        maskS2clouds
    )

    .select(
        ['B4','B8']
    )

)

    count=collection.size()
    
    if image_date:

     collection=collection.filterDate(

        image_date,

        ee.Date(
            image_date
        ).advance(
            1,
            'day'
        )
    )

    count=collection.size()

    img=ee.Image(

ee.Algorithms.If(

    count.gt(0),

    collection.median(),

    ee.Image.constant(
        [0,0]
    ).rename(
        ['B4','B8']
    )

))

    ndvi=img.normalizedDifference(
        ['B8','B4']
    ).rename(
        'NDVI'
    )

    return ndvi.clip(region)

      

@app.route('/')
def index(): return render_template_string(HTML)

@app.route('/api/countries')
def countries(): 
        return jsonify(ee.FeatureCollection("FAO/GAUL/2015/level0").aggregate_array('ADM0_NAME').sort().getInfo())

@app.route('/api/provinces')
def provinces():
        c = request.args.get('country')
        return jsonify({"provinces": ee.FeatureCollection("FAO/GAUL/2015/level1")
                        .filter(ee.Filter.eq('ADM0_NAME', c))
                        .aggregate_array('ADM1_NAME').sort().getInfo()})
@app.route('/api/province-boundary')
def province_boundary():

    country = request.args.get('country')
    province = request.args.get('province')

    fc = ee.FeatureCollection("FAO/GAUL/2015/level1")\
        .filter(ee.Filter.eq('ADM0_NAME', country))\
        .filter(ee.Filter.eq('ADM1_NAME', province))

    geojson = fc.geometry().getInfo()

    bounds = fc.geometry().bounds().getInfo()
    coords = bounds['coordinates'][0]

    xmin = coords[0][0]
    ymin = coords[0][1]
    xmax = coords[2][0]
    ymax = coords[2][1]

    return jsonify({
        "geojson": geojson,
        "bounds": [xmin, ymin, xmax, ymax]
    })      
def calculate_forest_area(region, year, month):

    ndvi = get_satellite_data(
        region,
        year,
        month
    )

    landcover = ee.Image(
        "ESA/WorldCover/v200/2021"
    ).select("Map")

    treeMask = landcover.eq(10)

    forestMask = ndvi.gt(0.7)\
        .And(treeMask)\
        .selfMask()

    area = forestMask.multiply(
        ee.Image.pixelArea()
    )

    result = area.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=region,
        scale=500,
        bestEffort=True,
        tileScale=16,
        maxPixels=1e13
    ).get('NDVI')

    if result:
        return result.getInfo()/10000

    return 0
@app.route('/api/analyze', methods=['POST'])
def analyze():

    data = request.json
    idx_type = data.get('type', 'NDVI')

    region = ee.FeatureCollection(
        "FAO/GAUL/2015/level1"
    )\
    .filter(
        ee.Filter.eq(
            'ADM0_NAME',
            data['country']
        )
    )\
    .filter(
        ee.Filter.inList(
            'ADM1_NAME',
            data['provinces']
        )
    )\
    .geometry()\
    .dissolve()

    ndvi_curr=get_satellite_data(

    region,

    data['year'],

    data['month'],

    data.get(
        'image_date'
    )

)

    landcover=ee.Image(
        "ESA/WorldCover/v200/2021"
    ).select("Map")

    treeMask=landcover.eq(10)

    forestMask=ndvi_curr.gt(0.7)\
        .And(treeMask)\
        .selfMask()

    if idx_type == 'BIOMASS':

        display_img = ndvi_curr.expression(
            '10.5 * exp(3.2 * b("NDVI"))'
        ).rename('val')

        vis = {
            'min':10,
            'max':150,
            'palette':[
                '#f7fcb9',
                '#addd8e',
                '#31a354',
                '#006837'
            ]
        }

        unit="Tấn/Ha"

    elif idx_type=='CO2':

        display_img = ndvi_curr.expression(
            '10.5 * exp(3.2 * b("NDVI")) *0.47*3.67'
        ).rename('val')

        vis = {
            'min':20,
            'max':250,
            'palette':[
                '#ebf8ff',
                '#90cdf4',
                '#3182ce',
                '#2c5282'
            ]
        }

        unit="Tấn/Ha"

    else:

        display_img = ndvi_curr.rename('val')
        vis = {
             'min':0.1,
             'max':0.8,
             'palette':[
             '#ff0000',
             '#ffff00',
             '#00ff00',
             '#006400'
            ]
        }

        unit="NDVI"

    # ===== NDVI 12 tháng =====

    months=list(range(1,13))

    images=[]

    for m in months:

        img=get_satellite_data(
            region,
            data['year'],
            str(m)
        )

        stat=img.reduceRegion(

            reducer=ee.Reducer.mean(),

            geometry=region,

            scale=1000,

            bestEffort=True,

            tileScale=16

        )

        images.append(

            ee.Feature(
                None,
                {
                    'month':m,
                    'NDVI':stat.get('NDVI')
                }
            )
        )

    fc=ee.FeatureCollection(
        images
    )

    raw_ndvi=fc.aggregate_array(
        'NDVI'
    ).getInfo()

    clean_ndvi=interpolate_data(
        raw_ndvi
    )

    biomass_series=[

        round(
            10.5*math.exp(3.2*v),
            2
        )

        if v>0 else 0

        for v in clean_ndvi
    ]

    co2_series=[

        round(
            b*0.47*3.67,
            2
        )

        for b in biomass_series
    ]

    # ===== diện tích hiện tại =====

    forestArea=forestMask.multiply(
        ee.Image.pixelArea()
    ).rename(
        'area'
    )

    area_info=forestArea.reduceRegion(

        reducer=ee.Reducer.sum(),

        geometry=region,

        scale=500,

        bestEffort=True,

        tileScale=16,

        maxPixels=1e13

    ).get('area')

    curr_area=0

    if area_info:

        curr_area=area_info.getInfo()/10000
    prev_year = int(data['year'])-1

    prev_area = calculate_forest_area(
    region,
    prev_year,
    data['month']
)

    change_percent = 0

    if prev_area>0:

     change_percent = round(
       ((curr_area-prev_area)/prev_area)*100,
        2
    )    
    avg_ndvi_val=round(
        max(clean_ndvi),
        3
    )

    current_stat_val=(
        avg_ndvi_val
        if idx_type=='NDVI'
        else max(biomass_series)
        if idx_type=='BIOMASS'
        else max(co2_series)
    )

    if avg_ndvi_val>0.7:

        assessment="đạt mức tốt"

    elif avg_ndvi_val>0.5:

        assessment="đạt mức trung bình"

    else:

        assessment="đạt mức thấp"

    trend="tăng" if change_percent>0 else "giảm"

    ai_comment=f"""
    Phân tích hoàn tất.<br>

    Chỉ số {idx_type} = {current_stat_val} {unit}<br>

    Diện tích rừng: {int(curr_area):,} ha<br>

    So với {prev_year}: {trend}
    {abs(change_percent)}%<br>

    {assessment}
    """
    forecast_data=predict_6months(

    region,

    data['year'],

    data['month']

)
    region_name = ", ".join(data["provinces"])

    decision_news = generate_ai_decision_news(
        region_name,
        data["year"],
        data["month"],
        {
            "forest_ha": int(curr_area),
            "forest_change": change_percent,
            "health_idx": avg_ndvi_val
        },
        forecast_data
    )
    display_img = display_img\
    .clip(region)\
    .unmask(0)

    bbox=region.bounds().getInfo()

    coords=bbox['coordinates'][0]

    xmin=coords[0][0]
    ymin=coords[0][1]

    xmax=coords[2][0]
    ymax=coords[2][1]

    return jsonify({

    "map":
    display_img.getMapId(
        vis
    )['tile_fetcher'].url_format,

    "bounds":[

        xmin,
        ymin,
        xmax,
        ymax

    ],

    "stats":{

        "forest_ha":
        int(curr_area),
         
        "forest_change":
        change_percent,

        "previous_year":
        prev_year,
        "current_val":
        current_stat_val,

        "unit":
        unit,

        "health_idx":
        avg_ndvi_val
    },

    "time_series":{

        "months":[
            f"T{m}"
            for m in range(
                1,
                13
            )
        ],

        "ndvi":
        clean_ndvi,

        "biomass":
        biomass_series,

        "co2":
        co2_series
    },
    "forecast":
    forecast_data,
    "decision_news":
    decision_news,
    "comment":
    ai_comment
})

    

@app.route('/api/pixel-history', methods=['POST'])
def pixel_history():

    data=request.json

    point=ee.Geometry.Point(
        [data['lng'],data['lat']]
    )

    img=(

        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )

        .filterBounds(point)

        .filterDate(
            f"{data['year']}-01-01",
            f"{data['year']}-12-31"
        )

        .filter(
            ee.Filter.lt(
                'CLOUDY_PIXEL_PERCENTAGE',
                70
            )
        )

        .map(maskS2clouds)

        .median()

    )
    ndvi = img.normalizedDifference(
    ['B8','B4']
).rename(
    'NDVI'
)

    val = ndvi.reduceRegion(

    reducer=ee.Reducer.mean(),

    geometry=point,

    scale=100,

    bestEffort=True

).get(
    'NDVI'
)

    if val:

     val=val.getInfo()

    else:

     val=0

    if val:

        biomass=round(
            10.5*math.exp(
                3.2*val
            ),
            2
        )

        co2=round(
            biomass*1.7249,
            2
        )

        return jsonify({

            "avg_ndvi":
            round(val,3),

            "avg_biomass":
            biomass,

            "avg_co2":
            co2

        })

    return jsonify({

        "avg_ndvi":0,
        "avg_biomass":0,
        "avg_co2":0

    })

if __name__ == '__main__':
        # Chạy trên port 5001
        app.run(debug=True, port=5001)





