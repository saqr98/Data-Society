import pandas as pd
countries = {'CZE', 'BGR', 'CYM', 'HRV', 'CHE', 'ISL', 'NZL', 'MDV', 'NIC', 'SLV', 'ARM', 'ABW', 'GRC', 'PRT', 'VAT', 'DNK', 'TTO', 'NER', 'BHS', 'JAM', 'MLT', 'TUR', 'IRL', 'IND', 'LCA', 'ZWE', 'BTN', 'BLR', 'MOZ', 'TKM', 'YEM', 'LVA', 'CHL', 'ETH', 'VEN', 'IDN', 'BRB', 'BEL', 'MAR', 'ARE', 'WLF', 'TUN', 'NLD', 'LSO', 'GIN', 'COD', 'SHN', 'COL', 'QAT', 'GRD', 'PAN', 'HTI', 'SWZ', 'PAK', 'MLI', 'LUX', 'BWA', 'SDN', 'AUT', 'RWA', 'CMR', 'UKR', 'VCT', 'KWT', 'MUS', 'GAB', 'ZMB', 'PRK', 'PSE', 'NGA', 'FJI', 'LBN', 'URY', 'BFA', 'DEU', 'PRY', 'GBR', 'COG', 'BGD', 'MCO', 'CHN', 'FIN', 'DOM', 'TON', 'GEO', 'TJK', 'VNM', 'LBY', 'SRB', 'NPL', 'MHL', 'CAN', 'UZB', 'ISR', 'FRA', 'OMN', 'BOL', 'GHA', 'AZE', 'LKA', 'RUS', 'GMB', 'DZA', 'KNA', 'WSM', 'AIA', 'BRA', 'MAC', 'USA', 'VUT', 'BMU', 'MMR', 'JOR', 'SAU', 'STP', 'BHR', 'CUB', 'IRQ', 'POL', 'MKD', 'DJI', 'SSD', 'ALB', 'LTU', 'PER', 'SOM', 'SLB', 'LBR', 'UGA', 'ITA', 'BRN', 'NAM', 'SMR', 'CIV', 'NRU', 'GTM', 'EST', 'MDA', 'ESP', 'ATG', 'SYC', 'CPV', 'LAO', 'LIE', 'MWI', 'ZAF', 'MDG', 'TGO', 'AGO', 'SWE', 'HND', 'PHL', 'COM', 'MEX', 'PLW', 'KGZ', 'FSM', 'KAZ', 'PNG', 'IRN', 'AUS', 'SVK', 'KEN', 'ECU', 'SUR', 'SEN', 'BLZ', 'COK', 'HUN', 'TUV', 'BDI', 'THA', 'GUY', 'EGY', 'AND', 'SYR', 'ERI', 'DMA', 'HKG', 'ARG', 'TZA', 'MYS', 'MRT', 'KIR', 'NOR', 'BEN', 'AFG', 'GNQ', 'KHM', 'SLE', 'KOR', 'CRI', 'MNG', 'SGP', 'GNB', 'JPN', 'CYP', 'CAF', 'TCD'}

if __name__ == '__main__':
    codes = pd.read_csv('Project/Code/data/countrycodes copy.csv',on_bad_lines='skip')
    longlat = pd.read_csv('Project/Code/data/countrylonglat.csv',on_bad_lines='skip')


    test = pd.merge(codes, longlat[['ISO-alpha3 code','Latitude','Longitude']], on='ISO-alpha3 code', how='left')
    test.to_csv('Project/Code/data/countrycodes_extended.csv', index=False)
    