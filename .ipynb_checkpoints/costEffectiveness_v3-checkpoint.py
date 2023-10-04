import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import Polygon, Point, MultiPoint
import math
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp
from numpy.random import rand, seed
from sklearn.cluster import KMeans 
pd.options.mode.chained_assignment = None  # default='warn'

def findFirstGuess(nHubs, candiHubs, matGrid): 
    '''
    selects hubs evenly spread throughout the study area. 
    inputs: 
        * nHubs 
        * candiHubs
        * matGrid 
    returns: 
        * hubs - gdf of selected hub locations with columns 'hubName' and 'geometry'
    '''
    # prep kMeans input 
    matGrid['x'] = matGrid.geometry.x
    matGrid['y'] = matGrid.geometry.y
    X = np.array(matGrid[['x', 'y']])

    # run kmeans 
    kmeans = KMeans(n_clusters=nHubs, random_state=0).fit(X)
    matGrid['cluster'] = kmeans.labels_

    # find centroid for each cluster and closest candiHubs point 
    clusterNums = list(matGrid.cluster.unique())
    candiIndexes = []
    for clusterNum in clusterNums: 

        # find centroid 
        clusterGdf = matGrid[matGrid.cluster == clusterNum]
        clusterCentroid = clusterGdf.dissolve().centroid

        # find closest candiHubs point
        candiIndex = candiHubs.geometry.sindex.nearest(clusterCentroid)[1,0]
        candiIndexes.append(candiIndex)

    # make hubs - gdf of chosen hubs 
    hubs = candiHubs.iloc[candiIndexes]
    
    # formatting
    hubs = hubs[['hubName', 'pPerSqm', 'geometry']]
    hubs = hubs.drop_duplicates() 
    # hubs = list(hubs.hubName)
    
    return hubs 

def assignHubsToGridCells(hubs, matGrid, distMatrix): 
    def findHub(matGrid_index): 
        hubs_index = [int(x) for x in list(hubs.hubName)]
        dists = distMatrix[matGrid_index, hubs_index]
        idxmin = np.argmin(dists)
        chosenHub = hubs_index[idxmin]
        return chosenHub
    matGrid['hubName'] = matGrid.index.map(lambda x: findHub(x))
    
    # formatting 
    matGrid = matGrid[['kgDemand', 'kgSupply', 'hubName', 'geometry']]
    
    return matGrid 

def calcTotCo2Reduction(hubs, matGridAss, tco2ReductionPerTon): 
    
    # calculate co2Avoided for each hub
    def calcCo2Avoided(row): 
        hubName = row.hubName

        # select hub's clients - both supply and demand
        clients = matGridAss[matGridAss.hubName == hubName]
        supply = clients[clients.kgSupply > 0][['kgSupply', 'geometry']]
        demand = clients[clients.kgDemand > 0][['kgDemand', 'geometry']]

        # calculate amount of supply and demand 
        supplyTons = supply.kgSupply.sum() / 1000
        demandTons = demand.kgDemand.sum() / 1000 * 0.5

        # calculate EOL emissions for primary and secondary scenario 
        co2Primary = tco2ReductionPerTon * supplyTons # all waste (supply) is incinerated, and not reused 
        wastedTons = supplyTons - demandTons if supplyTons - demandTons > 0 else 0 # supply is wasted if there is a surplus
        co2Secondary = tco2ReductionPerTon * wastedTons
        co2Avoided = co2Primary - co2Secondary

        return co2Avoided
    hubs['co2Avoided_tons'] = hubs.apply(lambda row: calcCo2Avoided(row), axis=1)

    # calculate CO2 avoided for all hubs
    totCo2Avoided = hubs.co2Avoided_tons.sum()

    return totCo2Avoided


def calcTotStorageCost(hubs, matGridAss, candiHubs): 

    def calcm2storage(row): 
        hubName = row.hubName 
        # print('\nhubName: {}'.format(hubName))

        # calculate material stored in hub in kg
        clients = matGridAss[matGridAss.hubName == hubName]
        kgSupply = clients.kgSupply.sum()
        kgDemand = clients.kgDemand.sum() * 0.5
        kgStored = kgSupply if kgDemand > kgSupply else kgDemand 
        kgStored = kgStored 

        # calculate kgStored assuming throughput 
        storageMonths = 6 # previously 6 months
        studyPeriodMonths = 5 * 12 # 5 years 
        percStorageMonths = storageMonths / studyPeriodMonths * 100 
        kgStored = kgStored * percStorageMonths / 100

        # calculate storage area required
        avTimberDensity = 510 # see CO2 opslag rekening tool
        avStorageHeight = 3.5 # meters 
        kgStoredPerM2 = avTimberDensity * 0.8 * avStorageHeight
        m2storage = kgStored / kgStoredPerM2

        # calculate storate area required + logistics space 
        percAreaLogistics = 30
        percAreaStorage = 100 - percAreaLogistics
        m2storage = m2storage / percAreaStorage * 100
        mStorageWidth = math.sqrt(m2storage)
        
        # min storage area 
        minStorageArea = 300
        if m2storage < minStorageArea: 
            m2storage = minStorageArea
        
        return m2storage
    
    def calcStoragePrice(row): 
        # calculate land and building price
        landPrice = row.pPerSqm
        buildingPrice = 382 # euros per sqm 
        totalPrice = landPrice + buildingPrice # euros per sqm 
        
        return totalPrice
    
    # sum storage cost for all hubs
    hubs['m2'] = hubs.apply(lambda row: calcm2storage(row), axis=1)
    hubs['storagePrice'] = hubs.apply(lambda row: calcStoragePrice(row), axis=1)
    hubs['storageCost'] = hubs.m2 * hubs.storagePrice
    totStorageCost = hubs.storageCost.sum()
    
    return totStorageCost


def calcTotTransportationCost(hubs, matGridAss, distMatrix, costPerKmPerTon):
    # define transportation price coefficient 
    # see data/transportation/tansCostPerKm_grootStukgoed.csv 
    # costPerKmPerTon = 0.92
    
    # calculate total transportation cost for all hubs 
    transCosts_allHubs = []
    for hubName in hubs.hubName: 
        hub = hubs[hubs.hubName == hubName]

        # find locations of demand and supply
        clients = matGridAss[matGridAss.hubName == hubName]
        demand = clients[clients.kgDemand > 0][['kgDemand', 'geometry']]
        supply = clients[clients.kgSupply > 0][['kgSupply', 'geometry']]

        # calculate total transportation cost for all client locations
        def calcTransCost(row): 
            distanceM = distMatrix[row.name, hubName]
            distanceKm = distanceM / 1000
            weightKg = row.iloc[0]
            weightTons = round(weightKg / 1000, 2)
            transCost = distanceKm * weightTons * costPerKmPerTon
            return transCost 
        demand['transCost'] = demand.apply(lambda row: calcTransCost(row), axis=1)
        supply['transCost'] = supply.apply(lambda row: calcTransCost(row), axis=1)

        totTransCost = demand.transCost.sum() + supply.transCost.sum()
        transCosts_allHubs.append(totTransCost)

    # print total transportation costs for all hubs 
    transCosts_allHubs = np.array(transCosts_allHubs).sum()

    return transCosts_allHubs 
    
    # sum transportation emissions all hubs
    hubs['transCost'] = hubs.hubName.map(lambda x: calcTransportationCost(x, transPriceCoef))
    totTransCost = hubs.transCost.sum()
    return totTransCost


def calcTotTransportationEmissions(hubs, matGridAss, distMatrix, emissionsPerTonPerKm): 
    
    # # define CO2 emissions per ton of material per km of transportation 
    # truckCapacityTons = 27 # same truck capacity as calculating transportation cost 'groot stukgoed' 
    # emissionsPerTruckPerKm_kg = 0.6653 # according to ecoInvent pdf p63 
    # emissionsPerTruckPerKm_tons = emissionsPerTruckPerKm_kg / 1000
    # emissionsPerTonPerKm = emissionsPerTruckPerKm_tons / truckCapacityTons
    # emissionsPerTonPerKm = 2.464074074074074e-05

    # calculate transportation emissions for each hub
    transEmissionsPerHub = []
    for hubName in list(hubs.hubName): 
        # selec hub's clients - both supply and demand 
        clients = matGridAss[matGridAss.hubName == hubName]
        supply = clients[clients.kgSupply > 0][['kgSupply', 'geometry']]
        demand = clients[clients.kgDemand > 0][['kgDemand', 'geometry']]

        # for each client, calculate:
        def calcTransEmissions(row): 
            distanceM = distMatrix[row.name, hubName]
            distanceKm = distanceM / 1000
            weightKg = row.iloc[0]
            weightTons = weightKg / 1000
            transEmissions = distanceKm * weightTons * emissionsPerTonPerKm
            return transEmissions 
        
        supply['travelEmissions'] = supply.apply(lambda row: calcTransEmissions(row), axis=1)
        demand['travelEmissions'] = demand.apply(lambda row: calcTransEmissions(row), axis=1)
        transEmissions_hub = supply.travelEmissions.sum() + demand.travelEmissions.sum()
        transEmissionsPerHub.append(transEmissions_hub)
    
    # sum transportation emissions for all hubs 
    totTransEmissions = np.array(transEmissionsPerHub).sum()
    
    return totTransEmissions


# -------------- READ REQUIRED FILES --------------
# read AMS data 
candiHubsAms = gpd.read_file('data/candiHubs_ams.shp')
matGridAms = gpd.read_file('data/matGrid_ams.shp')
distMatrixAms = np.load('data/costMatrix_ams.npy')

# packaging data into dataDicts 
data = {'matGrid': matGridAms, 'candiHubs': candiHubsAms, 'distMatrix': distMatrixAms}

# -------------- DEFINING COEFFICIENTS FOR COST EFFECTIVENESS CALCULATION --------------

# Transporation emissions coefficient: 
truckCapacityTons = 27 # same truck capacity as calculating transportation cost 'groot stukgoed' 
emissionsPerTruckPerKm_kg = 0.6653 # according to ecoInvent pdf p63 
emissionsPerTruckPerKm_tons = emissionsPerTruckPerKm_kg / 1000
emissionsPerTonPerKm_tons = emissionsPerTruckPerKm_tons / truckCapacityTons

# packaging coefficients into dictionary 
coefs = {
    'tco2ReductionPerTon': 1.52, 
    'storageMonths': 6,
    'buildingPrice': 382, 
    'costPerKmPerTon': 0.92, # trans cost (euros) - per km per ton
    'emissionsPerTonPerKm': emissionsPerTonPerKm_tons # trans emissions (tCO2) - per km per ton
}


def calcTotCostEffectiveness_r(pointsArray, reverse=False, runInR=True): 
    '''
    calculates cost effectiveness of a solution based on number of hubs. 
    inputs: 
        * `matGrid` - supply and demand of materials, from PBL dataset 
        * `candiHubs` - location of candidate sites of hubs, from IBIS dataset 
        * `distMatrix` - distance matrix between matGrid and candiHubs, for now euclidean distance
    returns: 
        * costEffeciveness
        * totStorageCost
        * totTransCost
        * totCo2Reduction 
        * totTransEmissions 
    '''
    # unpack data and coefs
    matGrid = data['matGrid']
    candiHubs = data['candiHubs']
    distMatrix = data['distMatrix']
       
    tco2ReductionPerTon = coefs['tco2ReductionPerTon']
    storageMonths = coefs['storageMonths']
    buildingPrice = coefs['buildingPrice']
    costPerKmPerTon = coefs['costPerKmPerTon']
    emissionsPerTonPerKm = coefs['emissionsPerTonPerKm']
    
    # make hubs gdf    
    hubNames = list(pointsArray[:, 0])
    if runInR: 
        hubNames = [int(x)-1 for x in hubNames]
    else: 
        hubNames = [int(x) for x in hubNames]
    if reverse:
        hubs = candiHubs[~candiHubs.hubName.isin(hubNames)]
    else: 
        hubs = candiHubs[candiHubs.hubName.isin(hubNames)]
                
    # assign hubs to grid cells 
    matGrid_hubsAssigned = assignHubsToGridCells(hubs, matGrid, distMatrix)

    # calculate sub-components
    totCo2Reduction = calcTotCo2Reduction(hubs, matGrid_hubsAssigned, tco2ReductionPerTon) 
    totStorageCost = calcTotStorageCost(hubs, matGrid_hubsAssigned, candiHubs)
    totTransCost = calcTotTransportationCost(hubs, matGrid_hubsAssigned, distMatrix, costPerKmPerTon)
    totTransEmissions = calcTotTransportationEmissions(hubs, matGrid_hubsAssigned, distMatrix, emissionsPerTonPerKm)

    # calculate cost effectiveness 
    costEffectiveness = (totStorageCost + totTransCost) / (totCo2Reduction - totTransEmissions)
    print(costEffectiveness)
    
    return costEffectiveness # , totStorageCost, totTransCost, totCo2Reduction, totTransEmissions  




