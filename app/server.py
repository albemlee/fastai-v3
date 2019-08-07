import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
import numpy as np
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles

export_file_url = 'https://drive.google.com/uc?export=download&id=1-wDJqSVUXvbFEgKULAnhHFF5rNzsg-F_'
export_file_name = 'resnet_18_121classes.pkl'

classes = ['buick_century', 'buick_enclave', 'buick_lacrosse', 'buick_lesabre', 'buick_lucerne', 'buick_parkavenue', 'buick_regal', 'buick_rendezvous', 'buick_verano', 'cadillac_cts', 'cadillac_escalade', 'cadillac_srx', 'cadillac_sts', 'chevrolet_avalanche', 'chevrolet_aveo', 'chevrolet_blazer', 'chevrolet_camaro', 'chevrolet_cavalier', 'chevrolet_cobalt', 'chevrolet_colorado', 'chevrolet_corvette', 'chevrolet_cruze', 'chevrolet_equinox', 'chevrolet_hhr', 'chevrolet_impala', 'chevrolet_malibu', 'chevrolet_montecarlo', 'chevrolet_s10', 'chevrolet_silverado', 'chevrolet_sonic', 'chevrolet_suburban', 'chevrolet_tahoe', 'chevrolet_trailblazer', 'chevrolet_traverse', 'chevrolet_uplander', 'chrysler_200', 'chrysler_300', 'chrysler_pacifica', 'chrysler_pt cruiser', 'chrysler_sebring', 'chrysler_town&country', 'dodge_avenger', 'dodge_caliber', 'dodge_challenger', 'dodge_charger', 'dodge_d250', 'dodge_dakota', 'dodge_dart', 'dodge_durango', 'dodge_grand caravan', 'dodge_journey', 'ford_bronco', 'ford_edge', 'ford_escape', 'ford_excursion', 'ford_expedition', 'ford_f150', 'ford_fiesta', 'ford_five hundred', 'ford_focus', 'ford_fusion', 'ford_mustang', 'ford_ranger', 'ford_taurus', 'ford_thunderbird', 'gmc_acadia', 'gmc_envoy', 'gmc_jimmy', 'gmc_sierra_2500', 'gmc_sierra_3500', 'gmc_yukon_1500', 'gmc_yukon_2500', 'honda_accord', 'honda_civic', 'honda_cr-v', 'honda_odyssey', 'honda_pilot', 'hyundai_elantra', 'hyundai_santafe', 'hyundai_sonata', 'jeep_compass', 'jeep_grand_cherokee', 'jeep_patriot', 'kia_soul', 'lincoln_mks', 'lincoln_mkx', 'lincoln_mkz', 'lincoln_navigator', 'lincoln_towncar', 'mercury_grandmarquis', 'mercury_mariner', 'mercury_mountaineer', 'mercury_sable', 'nissan_altima', 'nissan_frontier', 'nissan_maxima', 'nissan_murano', 'nissan_pathfinder', 'nissan_sentra', 'nissan_titan', 'pontiac_g6', 'pontiac_grandam', 'pontiac_grandprix', 'pontiac_montana', 'pontiac_torrent', 'pontiac_vibe', 'saturn_ion', 'saturn_outlook', 'saturn_vue', 'subaru_forester', 'subaru_outback', 'toyota_camry', 'toyota_corolla', 'toyota_highlander', 'toyota_prius', 'toyota_rav4', 'toyota_sienna', 'toyota_tacoma', 'toyota_tundra', 'volkswagen_jetta', 'volkswagen_passat']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    prediction = learn.predict(img)

    # create dictionary with all prediction scores
    prediction_probabilities = {
        'index': [],
        'class': [],
        'score': []
    }
    for index in range(len(prediction[2])):
        prediction_probabilities['index'].append(index)
        prediction_probabilities['class'].append(classes[index])
        prediction_probabilities['score'].append(float(prediction[2][index]))

    # get top scores and classes
    x = 10
    top_x_idx = np.argsort(prediction_probabilities['score'])[-x:]
    top_x = {}
    for i in top_x_idx:
        top_x[classes[i]] = prediction_probabilities['score'][i]

    return JSONResponse({'result': str(top_x)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
