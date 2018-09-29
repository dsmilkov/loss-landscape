'use strict';

const fanInDiv = document.querySelector('#fan-in');

async function onLoad() {
  const data = new MnistData();
  await data.load();
  console.log(data);
  const landscape = new Landscape();
  landscape.init(data);
  const fcFanIn = computeCharts(makeFCModel, landscape, 'unit');

  plotCharts(await fcFanIn, fanInDiv);
}

onLoad();
