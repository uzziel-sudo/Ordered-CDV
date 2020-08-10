import streamlit as st
import pandas as pd
import seaborn as sns
from collections import OrderedDict
import matplotlib.pyplot as plt

df = pd.read_csv("kobodata.csv")
df.columns = df.columns.str.replace('/', '_')
data = df.sort_values("IDENTIFICATION_County")
# Columns to drop
list_drop = ['start', 'end', 'deviceid', 'phonenumber','MOH_acknowledgement', 'version1', 'version2',
 'version001', 'version002', 'version003', 'version004', 'version005', 'version006', 'version007', 'version008',
 'version009', 'version010','version011','version012','version013','version014','__version__','_version_','_version__001',
 '_version__002','meta_instanceID','_uuid','_submission_time','_tags', '_notes', 'GPS','_GPS_altitude','_GPS_precision'
]
data.drop(list_drop, axis =1, inplace = True)
# Renaming the columns
new_cols = ['County','Sub-County','Facility','MFLCODE','mfl','ANCPROVINSION','ANC Register-MOH 405','ANC improvised register',
'Mother Child Booklet','DV105','ancsource_jan','DV106_ancjansource','ancreport_jan','DV106_ancjanreport',
'DV106_dancjan','DV106_pancjan','DV106_rxancjan','DV106_rancjan','DV106_khisanc','DV106_ancjan','DV106_ancerror','DV106_xancjan',
'DV106_ranckhis','ancsource_feb','febancsource','ancreport_feb','febancreport','dancfeb','pancfeb','rxancfeb','rancfeb','khisancfeb',
'ancfeb','ancerrorfeb','xancfeb','ranckhisfeb','ancsource_mar','marancsource','ancreport_mar','marancreport','dancmar','pancmar',
'rxancmar','rancmar','khisancmar','ancmar','ancerrormar','xancmar','ranckhismar','ImmunizationIndicator',
'DV204_1_Immunization Permanent Register - MOH 510','DV204_2_Immunization tally sheets MOH702',
'DV204_3_Mother child booklet','DV_205_What_is_the_to_report_in_MOH_710','penta3source_jan','immujansource','penta3report_jan',
'immujanreport','dimmujan','pimmujan','rximmujan','rimmujan','khispenta','pentajan','pentaerror','xpentakhis','rpentakhis',
'penta3source_feb','immufebsource','penta3report_feb','immufebreport','dimmufeb','pimmufeb','rximmufeb','rimmufeb','khispentafeb',
'pentafeb','pentaerrorfeb','xpentakhisfeb','rpentakhisfeb','penta3source_mar','immumarsource','penta3report_mar','immumarreport',
'dimmumar','pimmumar','rximmumar','rimmumar','khispentamar','pentamar','pentaerrormar','xpentakhismar','rpentakhismar',
'SkilledBirthDelivery','DV303_1_Maternity Register (MOH 333)','DV303_2_Improvised register','DV303_3_Birth Notification Record',
'DV304','sbdsource_jan','sbdjansource','sbdreport_jan','sbdjanreport','dsbdjan','psbdjan','xsbdjan','rsbdjan','khissbd','sbdjan',
'sbderror','xrsbdkhis','rsbdkhis','sbdsource_feb','sdbfebsource','sbdreport_feb','sbdfebreport','dsbdfeb','psbdfeb','xsbdfeb',
'rsbdfeb','khissbdfeb','sbdfeb','sbderrorfeb','rxsbdkhisfeb','rsbdkhisfeb','sbdsource_mar','sbdmarsource','sbdreport_mar',
'sbdmarreport','dsbdmar','psbdmar','xsbdmar','rsbdmar','khissbdmar','sbdmar','sbderrormar','rxsbdkhismar','rsbdkhismar',
'FamilyPlanningIndicator','DV404_1_fp_register','DV404_2_improvised_r','fbsource_jan','FPRECOUNTJANUARY','fbsource_feb',
'FPRECOUNTFEBRUARY','fbsource_mar','DV407Recount','fbreport_jan','DV410_DV410records','fbreport_feb','DV410_DV410BRECORD',
'fbreport_mar','DV410_DV410C2RECORDS','DV408FPServices','progestinjan','progestinjana','khisprogestin','dprojan',
'errorprojan','reprojan','xreprojan','dprojanx','errorprojanx','dprojanxj','errorprojanxj','oraljan','oraljana','khisoral',
'doraljan','errororaljan','reoraljan','xreoraljan','doraljanx','errororaljanx','doraljanxj','errororaljanxj','emergencyjan',
'emergencyjana','khisemergency','demergjan','erroremergjan','reemergjan','xreemergjan','demergjanx','erroremergjanx',
'demergjanxj','erroremergjanxj','injectionsjan','injectionsjana','khisinjections','dinjectionsjan','errorinjectionsjan',
'reinjectionsjan','xreinjectionsjan','dinjectionsjanx','errorinjectionsjanx','dinjectionsjanxj','errorinjectionsjanxj',
'iucdjan','iucdjana','khisiucd','diucdjan','erroriucdjan','reiucdjan','xreiucdjan','diucdjanx','erroriucdjanx','diucdjanxj',
'erroriucdjanxj','implantsjan','implantsjana','khisimplants','dimplantsjan','errorimplantsjan','reimplantsjan',
'xreimplantsjan','dimplantsjanx','errorimplantsjanx','dimplantsjanxj','errorimplantsjanxj','btljan','btljana','khisbtl',
'dbtljan','errorbtljan','rebtljan','xrebtljan','dbtljanx','errorbtljanx','dbtljanxj','errorbtljanxj','febprogestin',
'febprogestinb','khisprogestinfeb','dprofeb','errorprofeb','reprofeb','xreprofeb','dprofebx','errorprofebx','dprofebxj',
'errorprofebxj','feboral','feboralb','khisoralfeb','doralfeb','errororalfeb','reoralfeb','xreoralfeb','doralfebx',
'errororalfebx','doralfebxj','errororalfebxj','febemergency','febemergencyb','khisemergencyfeb','demergfeb','erroremergfeb',
'reemergfeb','xreemergfeb','demergfebx','erroremergfebx','demergfebxj','erroremergfebxj','febinjections','febinjectionsb',
'khisinjectionsfeb','dinjectionsfeb','errorinjectionsfeb','reinjectionsfeb','xreinjectionsfeb','dinjectionsfebx',
'errorinjectionsfebx','dinjectionsfebxj','errorinjectionsfebxj','febiucd','febiucdb','khisiucdfeb','diucdfeb',
'erroriucdfeb','reiucdfeb','xreiucdfeb','diucdfebx','erroriucdfebx','diucdfebxj','erroriucdfebxj','febimplants',
'febimplantsb','khisimplantsfeb','dimplantsfeb','errorimplantsfeb','reimplantsfeb','xreimplantsfeb',
'dimplantsfebx','errorimplantsfebx','dimplantsfebxj','errorimplantsfebxj','febbtl','febbtlb','khisbtlfeb','dbtlfeb',
'errorbtlfeb','rebtlfeb','xrebtlfeb','dbtlfebx','errorbtlfebx','dbtlfebxj','errorbtlfebxj','marprogestin','marprogestinc',
'khisprogestinmar','dpromar','errorpromar','repromar','xrepromar','dpromarx','errorpromarx','dpromarxj','errorpromarxj',
'maroral','maroralc','khisoralmar','doralmar','errororalmar','reoralmar','xreoralmar','doralmarx','errororalmarx',
'doralmarxj','errororalmarxj','maremergency','maremergencyc','khisemergencymar','demergmar','erroremergmar','reemergmar',
'xreemergmar','demergmarx','erroremergmarx','demergmarxj','erroremergmarxj','marinjections','marinjectionsc',
'khisinjectionsmar','dinjectionsmar','errorinjectionsmar','reinjectionsmar','xreinjectionsmar','dinjectionsmarx',
'errorinjectionsmarx','dinjectionsmarxj','errorinjectionsmarxj','mariucd','mariucdc','khisiucdmar','diucdmar',
'erroriucdmar','reiucdmar','xreiucdmar','diucdmarx','erroriucdmarx','diucdmarxj','erroriucdmarxj','marimplants',
'marimplantsc','khisimplantsmar','dimplantsmar','errorimplantsmar','reimplantsmar','xreimplantsmar','dimplantsmarx',
'errorimplantsmarx','dimplantsmarxj','errorimplantsmarxj','marbtl','marbtlc','khisbtlmar','dbtlmar1','errorbtlma',
'rebtlmar','xrebtlmar','dbtlmar1x','errorbtlmarx','dbtlmar1xj','errorbtlmarxj','SAF100','SAF101B','SAF102','SAF103a',
'SAF103b','SAF104','SAF105','SAF105A1','SAF105A1_001','SAF105A3','SAF105A4','SAF105B1','SAF105B2','SAF105B3','SAF105B4',
'SAF105C1','SAF105C2','SAF105C3','SAF105C4','SAF105D1','SAF105D2','SAF105D3','SAF105D4','SAF109','SAF110','SAF111','SAF112',
'SAF113','SAF114','SAF115','SAF116','SAF117','SAF118','SAF119','SAF120','SAF121','SAF122','SAF123','SAF124','SAF125',
'Comments','lat','lon']

data.columns = new_cols

pd.reset_option('mode.chained_assignment')
with pd.option_context('mode.chained_assignment', None):
	data['anc_sd'] = data[['DV106_ancjansource','febancsource','marancsource']].sum(axis=1)
	data['anc_ms'] = data[['DV106_ancjanreport','febancreport','marancreport']].sum(axis=1)
	data['anc_khis'] = data[['DV106_khisanc','khisancfeb','khisancmar']].sum(axis=1)
	data['penta3_sd'] = data[['immujansource','immufebsource','immumarsource']].sum(axis=1)
	data['penta3_ms'] = data[['immujanreport','immufebreport','immumarreport']].sum(axis=1)
	data['penta3_khis'] = data[['khispenta','khispentafeb','khispentamar']].sum(axis=1)
	data['sbd_sd'] = data[['sbdjansource','sdbfebsource','sbdmarsource']].sum(axis=1)
	data['sbd_ms'] = data[['sbdjanreport','sbdfebreport','sbdmarreport']].sum(axis=1)
	data['sbd_khis'] = data[['khissbd','khissbdfeb','khissbdmar']].sum(axis=1)
	data['anc_ssum'] = ((data['anc_ms']-data['anc_sd'])/data['anc_sd'])*100
	data['anc_skhis'] = ((data['anc_khis']-data['anc_sd'])/data['anc_sd'])*100
	data['penta3_ssum'] = ((data['penta3_ms']-data['penta3_sd'])/data['penta3_sd'])*100
	data['penta3_skhis'] = ((data['penta3_khis']-data['penta3_sd'])/data['penta3_sd'])*100
	data['sbd_ssum'] = ((data['sbd_ms']-data['sbd_sd'])/data['sbd_sd'])*100
	data['sbd_skhis'] = ((data['sbd_khis']-data['sbd_sd'])/data['sbd_sd'])*100
	#FB
	data['FP_sd'] = data[['FPRECOUNTJANUARY','FPRECOUNTFEBRUARY','DV407Recount']].sum(axis=1)
	data['FP_ms'] = data[['DV410_DV410records','DV410_DV410BRECORD','DV410_DV410C2RECORDS']].sum(axis=1)
	data['POP_sd'] = data[['progestinjan','febprogestin','marprogestin']].sum(axis=1)
	data['POP_ms'] = data[['progestinjana','febprogestinb','marprogestinc']].sum(axis=1)
	data['POP_khis'] = data[['khisprogestin','khisprogestinfeb','khisprogestinmar']].sum(axis=1)
	data['COC_sd'] = data[['oraljan','feboral','maroral']].sum(axis=1)
	data['COC_ms'] = data[['oraljana','feboralb','maroralc']].sum(axis=1)
	data['COC_khis'] = data[['khisoral','khisoralfeb','khisoralmar']].sum(axis=1)
	data['E Pills_sd'] = data[['emergencyjan','febemergency','maremergency']].sum(axis=1)
	data['E Pills_ms'] = data[['emergencyjana','febemergencyb','maremergencyc']].sum(axis=1)
	data['E Pills_khis'] = data[['khisemergency','khisemergencyfeb','khisemergencymar']].sum(axis=1)
	data['Injection_sd'] = data[['injectionsjan','febinjections','marinjections']].sum(axis=1)
	data['Injection_ms'] = data[['injectionsjana','febinjectionsb','marinjectionsc']].sum(axis=1)
	data['Injection_khis'] = data[['khisinjections','khisinjectionsfeb','khisinjectionsmar']].sum(axis=1)
	data['IUCD_sd'] = data[['iucdjan','febiucd','mariucd']].sum(axis=1)
	data['IUCD_ms'] = data[['iucdjana','febiucdb','mariucdc']].sum(axis=1)
	data['IUCD_khis'] = data[['khisiucd','khisiucdfeb','khisiucdmar']].sum(axis=1)
	data['Implant_sd'] = data[['implantsjan','febimplants','marimplants']].sum(axis=1)
	data['Implant_ms'] = data[['implantsjana','febimplantsb','marimplantsc']].sum(axis=1)
	data['Implant_khis'] = data[['khisimplants','khisimplantsfeb','khisimplantsmar']].sum(axis=1)
	data['BTL_sd'] = data[['btljan','febbtl','marbtl']].sum(axis=1)
	data['BTL_ms'] = data[['btljana','febbtlb','marbtlc']].sum(axis=1)
	data['BTL_khis'] = data[['khisbtl','khisbtlfeb','khisbtlmar']].sum(axis=1)
	data['FP_overall'] = ((data['FP_ms']-data['FP_sd'])/data['FP_sd'])*100
	data['POP_ssum'] = ((data['POP_ms']-data['POP_sd'])/data['POP_sd'])*100
	data['POP_skhis'] = ((data['POP_khis']-data['POP_sd'])/data['POP_sd'])*100
	data['COC_ssum'] = ((data['COC_ms']-data['COC_sd'])/data['COC_sd'])*100
	data['COC_skhis'] = ((data['COC_khis']-data['COC_sd'])/data['COC_sd'])*100
	data['E Pills_ssum'] = ((data['E Pills_ms']-data['E Pills_sd'])/data['E Pills_sd'])*100
	data['E Pills_skhis'] = ((data['E Pills_khis']-data['E Pills_sd'])/data['E Pills_sd'])*100
	data['Injection_ssum'] = ((data['Injection_ms']-data['Injection_sd'])/data['Injection_sd'])*100
	data['Injection_skhis'] = ((data['Injection_khis']-data['Injection_sd'])/data['Injection_sd'])*100
	data['IUCD_ssum'] = ((data['IUCD_ms']-data['IUCD_sd'])/data['IUCD_sd'])*100
	data['IUCD_skhis'] = ((data['IUCD_khis']-data['IUCD_sd'])/data['IUCD_sd'])*100
	data['Implant_ssum'] = ((data['Implant_ms']-data['Implant_sd'])/data['Implant_sd'])*100
	data['Implant_skhis'] = ((data['Implant_khis']-data['Implant_sd'])/data['Implant_sd'])*100
	data['BTL_ssum'] = ((data['BTL_ms']-data['BTL_sd'])/data['BTL_sd'])*100
	data['BTL_skhis'] = ((data['BTL_khis']-data['BTL_sd'])/data['BTL_sd'])*100

# Availability of documents used at the facility
anc_docs_df = data[['County','Sub-County','Facility','MFLCODE','mfl','ancsource_jan','ancsource_feb','ancsource_mar',
'ancreport_jan','ancreport_feb','ancreport_mar']]
anc_docs = data[['County','Sub-County','Facility','MFLCODE','mfl','ancsource_jan','ancsource_feb','ancsource_mar',
'ancreport_jan','ancreport_feb','ancreport_mar']]
penta3_docs = data[['County','Sub-County','Facility','MFLCODE','mfl','penta3source_jan','penta3source_feb',
'penta3source_mar','penta3report_jan','penta3report_feb','penta3report_mar']]
sbd_docs = data[['County','Sub-County','Facility','MFLCODE','mfl','sbdsource_jan','sbdsource_feb','sbdsource_mar',
'sbdreport_jan','sbdreport_feb','sbdreport_mar']]
fb_docs = data[['County','Sub-County','Facility','MFLCODE','mfl','fbsource_jan','fbsource_feb','fbsource_mar',
'fbreport_jan','fbreport_feb','fbreport_mar']]
###ANC
df1 = anc_docs.melt(id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
	value_vars =['ancsource_jan','ancsource_feb','ancsource_mar'],
	var_name='anc source',	value_name='source_A')
df2 = anc_docs.melt(id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
	value_vars =['ancreport_jan','ancreport_feb','ancreport_mar'],
	var_name='anc summary',	value_name='summary_A')

df1 = df1.set_index(['County','Sub-County','Facility','MFLCODE','mfl', df1.groupby(['County','Sub-County','Facility','MFLCODE','mfl']).cumcount()])
df2 = df2.set_index(['County','Sub-County','Facility','MFLCODE','mfl', df2.groupby(['County','Sub-County','Facility','MFLCODE','mfl']).cumcount()])

df3 = (pd.concat([df1,df2], axis=1)
	.sort_index(level=5)
	.reset_index(level=5, drop=True)
	.reset_index())
df4 = df3.melt(id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
	value_vars =['anc source','anc summary'],
	var_name='ssum',	value_name='ssum_A')
df5 = df3.melt(id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
	value_vars =['source_A','summary_A'],
	var_name='source available',	value_name='completeness')

df4 = df4.set_index(['County','Sub-County','Facility','MFLCODE','mfl', df4.groupby(['County','Sub-County','Facility','MFLCODE','mfl']).cumcount()])
df5 = df5.set_index(['County','Sub-County','Facility','MFLCODE','mfl', df5.groupby(['County','Sub-County','Facility','MFLCODE','mfl']).cumcount()])
df6 = (pd.concat([df4,df5], axis=1)
	.sort_index(level=5)
	.reset_index(level=5, drop=True)
	.reset_index())
df6['Availability'] = df6['completeness']

###Penta3

dfb1 = penta3_docs.melt(id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
	value_vars =['penta3source_jan','penta3source_feb','penta3source_mar'],
	var_name='penta3 source',	value_name='source_A')
dfb2 = penta3_docs.melt(id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
	value_vars =['penta3report_jan','penta3report_feb','penta3report_mar'],
	var_name='penta3 summary',	value_name='summary_A')

dfb1 = dfb1.set_index(['County','Sub-County','Facility','MFLCODE','mfl', dfb1.groupby(['County','Sub-County','Facility','MFLCODE','mfl']).cumcount()])
dfb2 = dfb2.set_index(['County','Sub-County','Facility','MFLCODE','mfl', dfb2.groupby(['County','Sub-County','Facility','MFLCODE','mfl']).cumcount()])

dfb3 = (pd.concat([dfb1,dfb2], axis=1)
	.sort_index(level=5)
	.reset_index(level=5, drop=True)
	.reset_index())
dfb4 = dfb3.melt(id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
	value_vars =['penta3 source','penta3 summary'],
	var_name='ssum',	value_name='ssum_A')
dfb5 = dfb3.melt(id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
	value_vars =['source_A','summary_A'],
	var_name='source available',	value_name='completeness')

dfb4 = dfb4.set_index(['County','Sub-County','Facility','MFLCODE','mfl', dfb4.groupby(['County','Sub-County','Facility','MFLCODE','mfl']).cumcount()])
dfb5 = dfb5.set_index(['County','Sub-County','Facility','MFLCODE','mfl', dfb5.groupby(['County','Sub-County','Facility','MFLCODE','mfl']).cumcount()])
dfb6 = (pd.concat([dfb4,dfb5], axis=1)
	.sort_index(level=5)
	.reset_index(level=5, drop=True)
	.reset_index())
dfb6['Availability'] = dfb6['completeness']

###SBD

dfc1 = sbd_docs.melt(id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
	value_vars =['sbdsource_jan','sbdsource_feb','sbdsource_mar'],
	var_name='sbd source',	value_name='source_A')
dfc2 = sbd_docs.melt(id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
	value_vars =['sbdreport_jan','sbdreport_feb','sbdreport_mar'],
	var_name='sbd summary',	value_name='summary_A')

dfc1 = dfc1.set_index(['County','Sub-County','Facility','MFLCODE','mfl', dfc1.groupby(['County','Sub-County','Facility','MFLCODE','mfl']).cumcount()])
dfc2 = dfc2.set_index(['County','Sub-County','Facility','MFLCODE','mfl', dfc2.groupby(['County','Sub-County','Facility','MFLCODE','mfl']).cumcount()])

dfc3 = (pd.concat([dfc1,dfc2], axis=1)
	.sort_index(level=5)
	.reset_index(level=5, drop=True)
	.reset_index())
dfc4 = dfc3.melt(id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
	value_vars =['sbd source','sbd summary'],
	var_name='ssum',	value_name='ssum_A')
dfc5 = dfc3.melt(id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
	value_vars =['source_A','summary_A'],
	var_name='source available',	value_name='completeness')

dfc4 = dfc4.set_index(['County','Sub-County','Facility','MFLCODE','mfl', dfc4.groupby(['County','Sub-County','Facility','MFLCODE','mfl']).cumcount()])
dfc5 = dfc5.set_index(['County','Sub-County','Facility','MFLCODE','mfl', dfc5.groupby(['County','Sub-County','Facility','MFLCODE','mfl']).cumcount()])
dfc6 = (pd.concat([dfc4,dfc5], axis=1)
	.sort_index(level=5)
	.reset_index(level=5, drop=True)
	.reset_index())
dfc6['Availability'] = dfc6['completeness']

###FB

dfd1 = fb_docs.melt(id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
	value_vars =['fbsource_jan','fbsource_feb','fbsource_mar'],
	var_name='fb source',	value_name='source_A')
dfd2 = fb_docs.melt(id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
	value_vars =['fbreport_jan','fbreport_feb','fbreport_mar'],
	var_name='fb summary',	value_name='summary_A')

dfd1 = dfd1.set_index(['County','Sub-County','Facility','MFLCODE','mfl', dfd1.groupby(['County','Sub-County','Facility','MFLCODE','mfl']).cumcount()])
dfd2 = dfd2.set_index(['County','Sub-County','Facility','MFLCODE','mfl', dfd2.groupby(['County','Sub-County','Facility','MFLCODE','mfl']).cumcount()])

dfd3 = (pd.concat([dfd1,dfd2], axis=1)
	.sort_index(level=5)
	.reset_index(level=5, drop=True)
	.reset_index())
dfd4 = dfd3.melt(id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
	value_vars =['fb source','fb summary'],
	var_name='ssum',	value_name='ssum_A')
dfd5 = dfd3.melt(id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
	value_vars =['source_A','summary_A'],
	var_name='source available',	value_name='completeness')

dfd4 = dfd4.set_index(['County','Sub-County','Facility','MFLCODE','mfl', dfd4.groupby(['County','Sub-County','Facility','MFLCODE','mfl']).cumcount()])
dfd5 = dfd5.set_index(['County','Sub-County','Facility','MFLCODE','mfl', dfd5.groupby(['County','Sub-County','Facility','MFLCODE','mfl']).cumcount()])
dfd6 = (pd.concat([dfd4,dfd5], axis=1)
	.sort_index(level=5)
	.reset_index(level=5, drop=True)
	.reset_index())
dfd6['Availability'] = dfd6['completeness']

def trans_Availability(x):
	if x == "yes__available":
		return 1
	elif x == "yes__available_1":
		return 1
	elif x == "yes__available_2":
		return 1
	elif x == "No":
		return 1
def trans_complete(x):
	if x == "yes__available":
		return "yes available and complete"
	elif x == "yes__available_1":
		return "yes available but partly recorded"
	elif x == "yes__available_2":
		return "yes available but no data recorded"
	

pd.reset_option('mode.chained_assignment')
with pd.option_context('mode.chained_assignment', None):
	df6['Availability'] = df6['Availability'].apply(trans_Availability)
	df6['completeness'] = df6['completeness'].apply(trans_complete)
	dfb6['Availability'] = dfb6['Availability'].apply(trans_Availability)
	dfb6['completeness'] = dfb6['completeness'].apply(trans_complete)
	dfc6['Availability'] = dfc6['Availability'].apply(trans_Availability)
	dfc6['completeness'] = dfc6['completeness'].apply(trans_complete)
	dfd6['Availability'] = dfd6['Availability'].apply(trans_Availability)
	dfd6['completeness'] = dfd6['completeness'].apply(trans_complete)


# Page of all 47 Counties
def Baringo_County():
	st.title("Baringo County")
	st.write("Below are Facilities covered")
	Baringo_data = data[data['County'] == 'Baringo County']
	Baringo_df6 = df6[df6['County'] == 'Baringo County']
	Baringo_dfb6 = dfb6[dfb6['County'] == 'Baringo County']
	Baringo_dfc6 = dfc6[dfc6['County'] == 'Baringo County']
	Baringo_dfd6 = dfd6[dfd6['County'] == 'Baringo County']
	st.write(Baringo_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Baringo_data['lat']),
		'longitude': min(Baringo_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Baringo_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Baringo_ssum = Baringo_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Baringo_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Baringo_fp_ssum = Baringo_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Baringo_fp_skhis = Baringo_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Baringo_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Baringo_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Baringo_data[['Facility','anc_ssum']][Baringo_data['anc_ssum'] >=-50]
		freq_dist1 = Baringo_data[['Facility','anc_ssum']][Baringo_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Baringo_data[['Facility','penta3_ssum']][Baringo_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Baringo_data[['Facility','sbd_ssum']][Baringo_data['sbd_ssum']>20]
		st.write(freq_dist)

	Baringo_skhis = Baringo_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Baringo_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	test = Baringo_df6['completeness'].value_counts()
	st.write(test)
	chart = sns.catplot(x="completeness", y="Availability", data=Baringo_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)
	#data = dataset.query("Accuracy > 50")
	


def Bomet_County():
	st.title("Baringo County")
	st.write("Below are Facilities covered")
	Bomet_data = data[data['County'] == 'Bomet County']
	Bomet_df6 = df6[df6['County'] == 'Bomet County']
	Bomet_dfb6 = dfb6[dfb6['County'] == 'Bomet County']
	Bomet_dfc6 = dfc6[dfc6['County'] == 'Bomet County']
	Bomet_dfd6 = dfd6[dfd6['County'] == 'Bomet County']
	st.write(Bomet_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Bomet_data['lat']),
		'longitude': min(Bomet_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Bomet_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Bomet_ssum = Bomet_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Bomet_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Bomet_fp_ssum = Bomet_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Bomet_fp_skhis = Bomet_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Bomet_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Bomet_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Bomet_data[['Facility','anc_ssum']][Bomet_data['anc_ssum'] >=-50]
		freq_dist1 = Bomet_data[['Facility','anc_ssum']][Bomet_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Bomet_data[['Facility','penta3_ssum']][Bomet_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Bomet_data[['Facility','sbd_ssum']][Bomet_data['sbd_ssum']>20]
		st.write(freq_dist)

	Bomet_skhis = Bomet_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Bomet_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Bomet_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)
	#data = data

def Bungoma_County():
	st.title("Bungoma County")
	st.write("Below are Facilities covered")
	Bungoma_data = data[data['County'] == 'Bungoma County']
	Bungoma_df6 = df6[df6['County'] == 'Bungoma County']
	Bungoma_dfb6 = dfb6[dfb6['County'] == 'Bungoma County']
	Bungoma_dfc6 = dfc6[dfc6['County'] == 'Bungoma County']
	Bungoma_dfd6 = dfd6[dfd6['County'] == 'Bungoma County']
	st.write(Bungoma_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Bungoma_data['lat']),
		'longitude': min(Bungoma_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Bungoma_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Bungoma_ssum = Bungoma_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Bungoma_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Bungoma_fp_ssum = Bungoma_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Bungoma_fp_skhis = Bungoma_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Bungoma_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Bungoma_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Bungoma_data[['Facility','anc_ssum']][Bungoma_data['anc_ssum'] >=-50]
		freq_dist1 = Bungoma_data[['Facility','anc_ssum']][Bungoma_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Bungoma_data[['Facility','penta3_ssum']][Bungoma_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Bungoma_data[['Facility','sbd_ssum']][Bungoma_data['sbd_ssum']>20]
		st.write(freq_dist)

	Bungoma_skhis = Bungoma_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Bungoma_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Bungoma_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Busia_County():
	st.title("Busia County")
	st.write("Below are Facilities covered")
	Busia_data = data[data['County'] == 'Busia County']
	Busia_df6 = df6[df6['County'] == 'Busia County']
	Busia_dfb6 = dfb6[dfb6['County'] == 'Busia County']
	Busia_dfc6 = dfc6[dfc6['County'] == 'Busia County']
	Busia_dfd6 = dfd6[dfd6['County'] == 'Busia County']
	st.write(Busia_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Busia_data['lat']),
		'longitude': min(Busia_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Busia_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Busia_ssum = Busia_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Busia_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Busia_fp_ssum = Busia_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Busia_fp_skhis = Busia_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Busia_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Busia_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Busia_data[['Facility','anc_ssum']][Busia_data['anc_ssum'] >=-50]
		freq_dist1 = Busia_data[['Facility','anc_ssum']][Busia_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Busia_data[['Facility','penta3_ssum']][Busia_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Busia_data[['Facility','sbd_ssum']][Busia_data['sbd_ssum']>20]
		st.write(freq_dist)

	Busia_skhis = Busia_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Busia_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Busia_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Elgeyo_Marakwet_County():
	st.title("Elgeyo-Marakwet County")
	st.write("Below are Facilities covered")
	Elgeyo_Marakwet_data = data[data['County'] == 'Elgeyo-Marakwet County']
	Elgeyo_Marakwet_df6 = df6[df6['County'] == 'Elgeyo-Marakwet County']
	Elgeyo_Marakwet_dfb6 = dfb6[dfb6['County'] == 'Elgeyo-Marakwet County']
	Elgeyo_Marakwet_dfc6 = dfc6[dfc6['County'] == 'Elgeyo-Marakwet County']
	Elgeyo_Marakwet_dfd6 = dfd6[dfd6['County'] == 'Elgeyo-Marakwet County']
	st.write(Elgeyo_Marakwet_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Elgeyo_Marakwet_data['lat']),
		'longitude': min(Elgeyo_Marakwet_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Elgeyo_Marakwet_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Elgeyo_Marakwet_ssum = Elgeyo_Marakwet_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Elgeyo_Marakwet_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Elgeyo_Marakwet_fp_ssum = Elgeyo_Marakwet_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Elgeyo_Marakwet_fp_skhis = Elgeyo_Marakwet_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Elgeyo_Marakwet_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Elgeyo_Marakwet_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Elgeyo_Marakwet_data[['Facility','anc_ssum']][Elgeyo_Marakwet_data['anc_ssum'] >=-50]
		freq_dist1 = Elgeyo_Marakwet_data[['Facility','anc_ssum']][Elgeyo_Marakwet_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Elgeyo_Marakwet_data[['Facility','penta3_ssum']][Elgeyo_Marakwet_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Elgeyo_Marakwet_data[['Facility','sbd_ssum']][Elgeyo_Marakwet_data['sbd_ssum']>20]
		st.write(freq_dist)

	Elgeyo_Marakwet_skhis = Elgeyo_Marakwet_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Elgeyo_Marakwet_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Elgeyo_Marakwet_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Embu_County():
	st.title("Embu County")
	st.write("Below are Facilities covered")
	Embu_data = data[data['County'] == 'Embu County']
	Embu_df6 = df6[df6['County'] == 'Embu County']
	Embu_dfb6 = dfb6[dfb6['County'] == 'Embu County']
	Embu_dfc6 = dfc6[dfc6['County'] == 'Embu County']
	Embu_dfd6 = dfd6[dfd6['County'] == 'Embu County']
	st.write(Embu_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Embu_data['lat']),
		'longitude': min(Embu_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Embu_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Embu_ssum = Embu_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Embu_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Embu_fp_ssum = Embu_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Embu_fp_skhis = Embu_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Embu_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Embu_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Embu_data[['Facility','anc_ssum']][Embu_data['anc_ssum'] >=-50]
		freq_dist1 = Embu_data[['Facility','anc_ssum']][Embu_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Embu_data[['Facility','penta3_ssum']][Embu_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Embu_data[['Facility','sbd_ssum']][Embu_data['sbd_ssum']>20]
		st.write(freq_dist)

	Embu_skhis = Embu_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Embu_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Embu_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Garissa_County():
	st.title("Garissa County")
	st.write("Below are Facilities covered")
	Garissa_data = data[data['County'] == 'Garissa County']
	Garissa_df6 = df6[df6['County'] == 'Garissa County']
	Garissa_dfb6 = dfb6[dfb6['County'] == 'Garissa County']
	Garissa_dfc6 = dfc6[dfc6['County'] == 'Garissa County']
	Garissa_dfd6 = dfd6[dfd6['County'] == 'Garissa County']
	st.write(Garissa_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Garissa_data['lat']),
		'longitude': min(Garissa_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Garissa_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Garissa_ssum = Garissa_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Garissa_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Garissa_fp_ssum = Garissa_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Garissa_fp_skhis = Garissa_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Garissa_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	st.markdown(
		'Hagarbul Dispensary is Under reporting by 100%')
	fp_skhis = pd.melt(Garissa_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Garissa_data[['Facility','anc_ssum']][Garissa_data['anc_ssum'] >=-50]
		freq_dist1 = Garissa_data[['Facility','anc_ssum']][Garissa_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Garissa_data[['Facility','penta3_ssum']][Garissa_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Garissa_data[['Facility','sbd_ssum']][Garissa_data['sbd_ssum']>50]
		st.write(freq_dist)

	Garissa_skhis = Garissa_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Garissa_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Garissa_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Homa_Bay_County():
	st.title("Homa Bay County")
	st.write("Below are Facilities covered")
	Homa_Bay_data = data[data['County'] == 'Homa Bay County']
	Homa_Bay_df6 = df6[df6['County'] == 'Homa Bay County']
	Homa_Bay_dfb6 = dfb6[dfb6['County'] == 'Homa Bay County']
	Homa_Bay_dfc6 = dfc6[dfc6['County'] == 'Homa Bay County']
	Homa_Bay_dfd6 = dfd6[dfd6['County'] == 'Homa Bay County']
	st.write(Homa_Bay_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Homa_Bay_data['lat']),
		'longitude': min(Homa_Bay_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Homa_Bay_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Homa_Bay_ssum = Homa_Bay_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Homa_Bay_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Homa_Bay_fp_ssum = Homa_Bay_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Homa_Bay_fp_skhis = Homa_Bay_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Homa_Bay_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Homa_Bay_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Homa_Bay_data[['Facility','anc_ssum']][Homa_Bay_data['anc_ssum'] >=-50]
		freq_dist1 = Homa_Bay_data[['Facility','anc_ssum']][Homa_Bay_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Homa_Bay_data[['Facility','penta3_ssum']][Homa_Bay_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Homa_Bay_data[['Facility','sbd_ssum']][Homa_Bay_data['sbd_ssum']>20]
		st.write(freq_dist)

	Homa_Bay_skhis = Homa_Bay_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Homa_Bay_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Homa_Bay_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Isiolo_County():
	st.title("Isiolo County")
	st.write("Below are Facilities covered")
	Isiolo_data = data[data['County'] == 'Isiolo County']
	Isiolo_df6 = df6[df6['County'] == 'Isiolo County']
	Isiolo_dfb6 = dfb6[dfb6['County'] == 'Isiolo County']
	Isiolo_dfc6 = dfc6[dfc6['County'] == 'Isiolo County']
	Isiolo_dfd6 = dfd6[dfd6['County'] == 'Isiolo County']
	st.write(Isiolo_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Isiolo_data['lat']),
		'longitude': min(Isiolo_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Isiolo_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Isiolo_ssum = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Isiolo_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Isiolo_fp_ssum = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Isiolo_fp_skhis = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Isiolo_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Isiolo_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Isiolo_data[['Facility','anc_ssum']][Isiolo_data['anc_ssum'] >=-50]
		freq_dist1 = Isiolo_data[['Facility','anc_ssum']][Isiolo_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Isiolo_data[['Facility','penta3_ssum']][Isiolo_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Isiolo_data[['Facility','sbd_ssum']][Isiolo_data['sbd_ssum']>20]
		st.write(freq_dist)

	Isiolo_skhis = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Isiolo_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Isiolo_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Kajiado_County():
	st.title("Kajiado County")
	st.write("Below are Facilities covered")
	Kajiado_data = data[data['County'] == 'Kajiado County']
	Kajiado_df6 = df6[df6['County'] == 'Kajiado County']
	Kajiado_dfb6 = dfb6[dfb6['County'] == 'Kajiado County']
	Kajiado_dfc6 = dfc6[dfc6['County'] == 'Kajiado County']
	Kajiado_dfd6 = dfd6[dfd6['County'] == 'Kajiado County']
	st.write(Kajiado_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Kajiado_data['lat']),
		'longitude': min(Kajiado_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Kajiado_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Kajiado_ssum = Kajiado_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Kajiado_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Kajiado_fp_ssum = Kajiado_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Kajiado_fp_skhis = Kajiado_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Kajiado_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Kajiado_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Kajiado_data[['Facility','anc_ssum']][Kajiado_data['anc_ssum'] >=-50]
		freq_dist1 = Kajiado_data[['Facility','anc_ssum']][Kajiado_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Kajiado_data[['Facility','penta3_ssum']][Kajiado_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Kajiado_data[['Facility','sbd_ssum']][Kajiado_data['sbd_ssum']>20]
		st.write(freq_dist)

	Kajiado_skhis = Kajiado_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Kajiado_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Kajiado_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Kakamega_County():
	st.title("Kakamega County")
	st.write("Below are Facilities covered")
	Kakamega_data = data[data['County'] == 'Kakamega County']
	Kakamega_df6 = df6[df6['County'] == 'Kakamega County']
	Kakamega_dfb6 = dfb6[dfb6['County'] == 'Kakamega County']
	Kakamega_dfc6 = dfc6[dfc6['County'] == 'Kakamega County']
	Kakamega_dfd6 = dfd6[dfd6['County'] == 'Kakamega County']
	st.write(Kakamega_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Kakamega_data['lat']),
		'longitude': min(Kakamega_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Kakamega_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Kakamega_ssum = Kakamega_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Kakamega_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Kakamega_fp_ssum = Kakamega_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Kakamega_fp_skhis = Kakamega_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Kakamega_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Kakamega_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Kakamega_data[['Facility','anc_ssum']][Kakamega_data['anc_ssum'] >=-50]
		freq_dist1 = Kakamega_data[['Facility','anc_ssum']][Kakamega_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Kakamega_data[['Facility','penta3_ssum']][Kakamega_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Kakamega_data[['Facility','sbd_ssum']][Kakamega_data['sbd_ssum']>20]
		st.write(freq_dist)

	Kakamega_skhis = Kakamega_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Kakamega_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Kakamega_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)
def Kericho_County():
	st.title("Kericho County")
	st.write("Below are Facilities covered")
	Kericho_data = data[data['County'] == 'Kericho County']
	Kericho_df6 = df6[df6['County'] == 'Kericho County']
	Kericho_dfb6 = dfb6[dfb6['County'] == 'Kericho County']
	Kericho_dfc6 = dfc6[dfc6['County'] == 'Kericho County']
	Kericho_dfd6 = dfd6[dfd6['County'] == 'Kericho County']
	st.write(Kericho_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Kericho_data['lat']),
		'longitude': min(Kericho_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Kericho_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Kericho_ssum = Kericho_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Kericho_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Kericho_fp_ssum = Kericho_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Kericho_fp_skhis = Kericho_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Kericho_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Kericho_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Kericho_data[['Facility','anc_ssum']][Kericho_data['anc_ssum'] >=-50]
		freq_dist1 = Kericho_data[['Facility','anc_ssum']][Kericho_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Kericho_data[['Facility','penta3_ssum']][Kericho_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Kericho_data[['Facility','sbd_ssum']][Kericho_data['sbd_ssum']>20]
		st.write(freq_dist)

	Kericho_skhis = Kericho_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Kericho_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Kericho_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)
def Kiambu_County():
	st.title("Kiambu County")
	st.write("Below are Facilities covered")
	Kiambu_data = data[data['County'] == 'Kiambu County']
	Kiambu_df6 = df6[df6['County'] == 'Kiambu County']
	Kiambu_dfb6 = dfb6[dfb6['County'] == 'Kiambu County']
	Kiambu_dfc6 = dfc6[dfc6['County'] == 'Kiambu County']
	Kiambu_dfd6 = dfd6[dfd6['County'] == 'Kiambu County']
	st.write(Kiambu_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Kiambu_data['lat']),
		'longitude': min(Kiambu_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Kiambu_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Kiambu_ssum = Kiambu_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Kiambu_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Kiambu_fp_ssum = Kiambu_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Kiambu_fp_skhis = Kiambu_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Kiambu_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Kiambu_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Kiambu_data[['Facility','anc_ssum']][Kiambu_data['anc_ssum'] >=-50]
		freq_dist1 = Kiambu_data[['Facility','anc_ssum']][Kiambu_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Kiambu_data[['Facility','penta3_ssum']][Kiambu_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Kiambu_data[['Facility','sbd_ssum']][Kiambu_data['sbd_ssum']>20]
		st.write(freq_dist)

	Kiambu_skhis = Kiambu_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Kiambu_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Kiambu_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Kilifi_County():
	st.title("Kilifi County")
	st.write("Below are Facilities covered")
	Kilifi_data = data[data['County'] == 'Kilifi County']
	Kilifi_df6 = df6[df6['County'] == 'Kilifi County']
	Kilifi_dfb6 = dfb6[dfb6['County'] == 'Kilifi County']
	Kilifi_dfc6 = dfc6[dfc6['County'] == 'Kilifi County']
	Kilifi_dfd6 = dfd6[dfd6['County'] == 'Kilifi County']
	st.write(Kilifi_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Kilifi_data['lat']),
		'longitude': min(Kilifi_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Kilifi_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Kilifi_ssum = Kilifi_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Kilifi_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Kilifi_fp_ssum = Kilifi_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Kilifi_fp_skhis = Kilifi_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Kilifi_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Kilifi_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Kilifi_data[['Facility','anc_ssum']][Kilifi_data['anc_ssum'] >=-50]
		freq_dist1 = Kilifi_data[['Facility','anc_ssum']][Kilifi_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Kilifi_data[['Facility','penta3_ssum']][Kilifi_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Kilifi_data[['Facility','sbd_ssum']][Kilifi_data['sbd_ssum']>20]
		st.write(freq_dist)

	Kilifi_skhis = Kilifi_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Kilifi_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Kilifi_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Kirinyaga_County():
	st.title("Kirinyaga County")
	st.write("Below are Facilities covered")
	Kirinyaga_data = data[data['County'] == 'Kirinyaga County']
	Kirinyaga_df6 = df6[df6['County'] == 'Kirinyaga County']
	Kirinyaga_dfb6 = dfb6[dfb6['County'] == 'Kirinyaga County']
	Kirinyaga_dfc6 = dfc6[dfc6['County'] == 'Kirinyaga County']
	Kirinyaga_dfd6 = dfd6[dfd6['County'] == 'Kirinyaga County']
	st.write(Kirinyaga_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Kirinyaga_data['lat']),
		'longitude': min(Kirinyaga_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Kirinyaga_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Kirinyaga_ssum = Kirinyaga_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Kirinyaga_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Kirinyaga_fp_ssum = Kirinyaga_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Kirinyaga_fp_skhis = Kirinyaga_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Kirinyaga_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Kirinyaga_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Kirinyaga_data[['Facility','anc_ssum']][Kirinyaga_data['anc_ssum'] >=-50]
		freq_dist1 = Kirinyaga_data[['Facility','anc_ssum']][Kirinyaga_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Kirinyaga_data[['Facility','penta3_ssum']][Kirinyaga_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Kirinyaga_data[['Facility','sbd_ssum']][Kirinyaga_data['sbd_ssum']>20]
		st.write(freq_dist)

	Kirinyaga_skhis = Kirinyaga_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Kirinyaga_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Kirinyaga_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Kisii_County():
	st.title("Kisii County")
	st.write("Below are Facilities covered")
	Kisii_data = data[data['County'] == 'Kisii County']
	Kisii_df6 = df6[df6['County'] == 'Kisii County']
	Kisii_dfb6 = dfb6[dfb6['County'] == 'Kisii County']
	Kisii_dfc6 = dfc6[dfc6['County'] == 'Kisii County']
	Kisii_dfd6 = dfd6[dfd6['County'] == 'Kisii County']
	st.write(Kisii_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Kisii_data['lat']),
		'longitude': min(Kisii_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Kisii_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Kisii_ssum = Kisii_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Kisii_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Kisii_fp_ssum = Kisii_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Kisii_fp_skhis = Kisii_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Kisii_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Kisii_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Kisii_data[['Facility','anc_ssum']][Kisii_data['anc_ssum'] >=-50]
		freq_dist1 = Kisii_data[['Facility','anc_ssum']][Kisii_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Kisii_data[['Facility','penta3_ssum']][Kisii_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Kisii_data[['Facility','sbd_ssum']][Kisii_data['sbd_ssum']>20]
		st.write(freq_dist)

	Kisii_skhis = Kisii_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Kisii_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Kisii_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Kisumu_County():
	st.title("Kisumu County")
	st.write("Below are Facilities covered")
	Kisumu_data = data[data['County'] == 'Kisumu County']
	Kisumu_df6 = df6[df6['County'] == 'Kisumu County']
	Kisumu_dfb6 = dfb6[dfb6['County'] == 'Kisumu County']
	Kisumu_dfc6 = dfc6[dfc6['County'] == 'Kisumu County']
	Kisumu_dfd6 = dfd6[dfd6['County'] == 'Kisumu County']
	st.write(Kisumu_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Kisumu_data['lat']),
		'longitude': min(Kisumu_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Kisumu_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Kisumu_ssum = Kisumu_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Kisumu_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Kisumu_fp_ssum = Kisumu_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Kisumu_fp_skhis = Kisumu_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Kisumu_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Kisumu_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Kisumu_data[['Facility','anc_ssum']][Kisumu_data['anc_ssum'] >=-50]
		freq_dist1 = Kisumu_data[['Facility','anc_ssum']][Kisumu_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Kisumu_data[['Facility','penta3_ssum']][Kisumu_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Kisumu_data[['Facility','sbd_ssum']][Kisumu_data['sbd_ssum']>20]
		st.write(freq_dist)

	Kisumu_skhis = Kisumu_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Kisumu_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Kisumu_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Kitui_County():
	st.title("Kitui County")
	st.write("Below are Facilities covered")
	Kitui_data = data[data['County'] == 'Kitui County']
	Kitui_df6 = df6[df6['County'] == 'Kitui County']
	Kitui_dfb6 = dfb6[dfb6['County'] == 'Kitui County']
	Kitui_dfc6 = dfc6[dfc6['County'] == 'Kitui County']
	Kitui_dfd6 = dfd6[dfd6['County'] == 'Kitui County']
	st.write(Kitui_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Kitui_data['lat']),
		'longitude': min(Kitui_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Kitui_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Kitui_ssum = Kitui_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Kitui_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Kitui_fp_ssum = Kitui_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Kitui_fp_skhis = Kitui_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Kitui_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Kitui_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Kitui_data[['Facility','anc_ssum']][Kitui_data['anc_ssum'] >=-50]
		freq_dist1 = Kitui_data[['Facility','anc_ssum']][Kitui_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Kitui_data[['Facility','penta3_ssum']][Kitui_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Kitui_data[['Facility','sbd_ssum']][Kitui_data['sbd_ssum']>20]
		st.write(freq_dist)

	Kitui_skhis = Kitui_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Kitui_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Kitui_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Kwale_County():
	st.title("Kwale County")
	st.write("Below are Facilities covered")
	Kwale_data = data[data['County'] == 'Kwale County']
	Kwale_df6 = df6[df6['County'] == 'Kwale County']
	Kwale_dfb6 = dfb6[dfb6['County'] == 'Kwale County']
	Kwale_dfc6 = dfc6[dfc6['County'] == 'Kwale County']
	Kwale_dfd6 = dfd6[dfd6['County'] == 'Kwale County']
	st.write(Kwale_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Kwale_data['lat']),
		'longitude': min(Kwale_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Kwale_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Kwale_ssum = Kwale_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Kwale_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Kwale_fp_ssum = Kwale_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Kwale_fp_skhis = Kwale_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Kwale_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Kwale_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Kwale_data[['Facility','anc_ssum']][Kwale_data['anc_ssum'] >=-50]
		freq_dist1 = Kwale_data[['Facility','anc_ssum']][Kwale_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Kwale_data[['Facility','penta3_ssum']][Kwale_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Kwale_data[['Facility','sbd_ssum']][Kwale_data['sbd_ssum']>20]
		st.write(freq_dist)

	Kwale_skhis = Kwale_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Kwale_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Kwale_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Laikipia_County():
	st.title("Laikipia County")
	st.write("Below are Facilities covered")
	Laikipia_data = data[data['County'] == 'Laikipia County']
	Laikipia_df6 = df6[df6['County'] == 'Laikipia County']
	Laikipia_dfb6 = dfb6[dfb6['County'] == 'Laikipia County']
	Laikipia_dfc6 = dfc6[dfc6['County'] == 'Laikipia County']
	Laikipia_dfd6 = dfd6[dfd6['County'] == 'Laikipia County']
	st.write(Laikipia_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Laikipia_data['lat']),
		'longitude': min(Laikipia_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Laikipia_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Laikipia_ssum = Laikipia_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Laikipia_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Laikipia_fp_ssum = Laikipia_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Laikipia_fp_skhis = Laikipia_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Laikipia_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Laikipia_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Laikipia_data[['Facility','anc_ssum']][Laikipia_data['anc_ssum'] >=-50]
		freq_dist1 = Laikipia_data[['Facility','anc_ssum']][Laikipia_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Laikipia_data[['Facility','penta3_ssum']][Laikipia_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Laikipia_data[['Facility','sbd_ssum']][Laikipia_data['sbd_ssum']>20]
		st.write(freq_dist)

	Laikipia_skhis = Laikipia_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Laikipia_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Laikipia_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Lamu_County():
	st.title("Lamu County")
	st.write("Below are Facilities covered")
	Lamu_data = data[data['County'] == 'Lamu County']
	Lamu_df6 = df6[df6['County'] == 'Lamu County']
	Lamu_dfb6 = dfb6[dfb6['County'] == 'Lamu County']
	Lamu_dfc6 = dfc6[dfc6['County'] == 'Lamu County']
	Lamu_dfd6 = dfd6[dfd6['County'] == 'Lamu County']
	st.write(Lamu_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Lamu_data['lat']),
		'longitude': min(Lamu_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Lamu_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Lamu_ssum = Lamu_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Lamu_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Lamu_fp_ssum = Lamu_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Lamu_fp_skhis = Lamu_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Lamu_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Lamu_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = ILamudata[['Facility','anc_ssum']][Lamu_data['anc_ssum'] >=-50]
		freq_dist1 = Lamu_data[['Facility','anc_ssum']][Lamu_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Lamu_data[['Facility','penta3_ssum']][Lamu_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Lamu_data[['Facility','sbd_ssum']][Lamu_data['sbd_ssum']>20]
		st.write(freq_dist)

	Lamu_skhis = Lamu_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Lamu_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Lamu_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Machakos_County():
	st.title("Machakos County")
	st.write("Below are Facilities covered")
	Machakos_data = data[data['County'] == 'Machakos County']
	Machakos_df6 = df6[df6['County'] == 'Machakos County']
	Machakos_dfb6 = dfb6[dfb6['County'] == 'Machakos County']
	Machakos_dfc6 = dfc6[dfc6['County'] == 'Machakos County']
	Machakos_dfd6 = dfd6[dfd6['County'] == 'Machakos County']
	st.write(Machakos_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Machakos_data['lat']),
		'longitude': min(Machakos_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Machakos_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Machakos_ssum = Machakos_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Machakos_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Machakos_fp_ssum = Machakos_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Machakos_fp_skhis = Machakos_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Machakos_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Machakos_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Machakos_data[['Facility','anc_ssum']][Machakos_data['anc_ssum'] >=-50]
		freq_dist1 = Machakos_data[['Facility','anc_ssum']][Machakos_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Machakos_data[['Facility','penta3_ssum']][Machakos_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Machakos_data[['Facility','sbd_ssum']][Machakos_data['sbd_ssum']>20]
		st.write(freq_dist)

	Machakos_skhis = Machakos_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Machakos_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Machakos_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Makueni_County():
	st.title("Makueni County")
	st.write("Below are Facilities covered")
	Makueni_data = data[data['County'] == 'Makueni County']
	Makueni_df6 = df6[df6['County'] == 'Makueni County']
	Makueni_dfb6 = dfb6[dfb6['County'] == 'Makueni County']
	Makueni_dfc6 = dfc6[dfc6['County'] == 'Makueni County']
	Makueni_dfd6 = dfd6[dfd6['County'] == 'Makueni County']
	st.write(Makueni_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Makueni_data['lat']),
		'longitude': min(Makueni_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Makueni_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Makueni_ssum = Makueni_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Makueni_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Makueni_fp_ssum = Makueni_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Makueni_fp_skhis = Makueni_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Makueni_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Makueni_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Makueni_data[['Facility','anc_ssum']][Makueni_data['anc_ssum'] >=-50]
		freq_dist1 = Makueni_data[['Facility','anc_ssum']][Makueni_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Makueni_data[['Facility','penta3_ssum']][Makueni_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Makueni_data[['Facility','sbd_ssum']][Makueni_data['sbd_ssum']>20]
		st.write(freq_dist)

	Makueni_skhis = Makueni_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Makueni_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Makueni_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Mandera_County():
	st.title("Mandera County")
	st.write("Below are Facilities covered")
	Mandera_data = data[data['County'] == 'Mandera County']
	Mandera_df6 = df6[df6['County'] == 'Mandera County']
	Mandera_dfb6 = dfb6[dfb6['County'] == 'Mandera County']
	Mandera_dfc6 = dfc6[dfc6['County'] == 'Mandera County']
	Mandera_dfd6 = dfd6[dfd6['County'] == 'Mandera County']
	st.write(Mandera_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Mandera_data['lat']),
		'longitude': min(Mandera_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Mandera_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Mandera_ssum = Mandera_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Mandera_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Mandera_fp_ssum = Mandera_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Mandera_fp_skhis = Mandera_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Mandera_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Mandera_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Mandera_data[['Facility','anc_ssum']][Mandera_data['anc_ssum'] >=-50]
		freq_dist1 = Mandera_data[['Facility','anc_ssum']][Mandera_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Mandera_data[['Facility','penta3_ssum']][Mandera_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Mandera_data[['Facility','sbd_ssum']][Mandera_data['sbd_ssum']>20]
		st.write(freq_dist)

	Mandera_skhis = Mandera_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Mandera_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Mandera_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Marsabit_County():
	st.title("Marsabit County")
	st.write("Below are Facilities covered")
	Marsabit_data = data[data['County'] == 'Marsabit County']
	Marsabit_df6 = df6[df6['County'] == 'Marsabit County']
	Marsabit_dfb6 = dfb6[dfb6['County'] == 'Marsabit County']
	Marsabit_dfc6 = dfc6[dfc6['County'] == 'Marsabit County']
	Marsabit_dfd6 = dfd6[dfd6['County'] == 'Marsabit County']
	st.write(Marsabit_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Marsabit_data['lat']),
		'longitude': min(Marsabit_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Marsabit_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Marsabit_ssum = Marsabit_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Marsabit_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Marsabit_fp_ssum = Marsabit_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Marsabit_fp_skhis = Marsabit_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Marsabit_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Marsabit_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Marsabit_data[['Facility','anc_ssum']][Marsabit_data['anc_ssum'] >=-50]
		freq_dist1 = IMarsabitdata[['Facility','anc_ssum']][Marsabit_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Marsabit_data[['Facility','penta3_ssum']][Marsabit_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Marsabit_data[['Facility','sbd_ssum']][Marsabit_data['sbd_ssum']>20]
		st.write(freq_dist)

	Marsabit_skhis = Marsabit_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Marsabit_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Marsabit_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Meru_County():
	st.title("Isiolo County")
	st.write("Below are Facilities covered")
	Isiolo_data = data[data['County'] == 'Isiolo County']
	Isiolo_df6 = df6[df6['County'] == 'Isiolo County']
	Isiolo_dfb6 = dfb6[dfb6['County'] == 'Isiolo County']
	Isiolo_dfc6 = dfc6[dfc6['County'] == 'Isiolo County']
	Isiolo_dfd6 = dfd6[dfd6['County'] == 'Isiolo County']
	st.write(Isiolo_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Isiolo_data['lat']),
		'longitude': min(Isiolo_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Isiolo_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Isiolo_ssum = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Isiolo_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Isiolo_fp_ssum = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Isiolo_fp_skhis = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Isiolo_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Isiolo_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Isiolo_data[['Facility','anc_ssum']][Isiolo_data['anc_ssum'] >=-50]
		freq_dist1 = Isiolo_data[['Facility','anc_ssum']][Isiolo_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Isiolo_data[['Facility','penta3_ssum']][Isiolo_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Isiolo_data[['Facility','sbd_ssum']][Isiolo_data['sbd_ssum']>20]
		st.write(freq_dist)

	Isiolo_skhis = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Isiolo_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Isiolo_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Migori_County():
	st.title("Isiolo County")
	st.write("Below are Facilities covered")
	Isiolo_data = data[data['County'] == 'Isiolo County']
	Isiolo_df6 = df6[df6['County'] == 'Isiolo County']
	Isiolo_dfb6 = dfb6[dfb6['County'] == 'Isiolo County']
	Isiolo_dfc6 = dfc6[dfc6['County'] == 'Isiolo County']
	Isiolo_dfd6 = dfd6[dfd6['County'] == 'Isiolo County']
	st.write(Isiolo_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Isiolo_data['lat']),
		'longitude': min(Isiolo_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Isiolo_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Isiolo_ssum = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Isiolo_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Isiolo_fp_ssum = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Isiolo_fp_skhis = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Isiolo_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Isiolo_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Isiolo_data[['Facility','anc_ssum']][Isiolo_data['anc_ssum'] >=-50]
		freq_dist1 = Isiolo_data[['Facility','anc_ssum']][Isiolo_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Isiolo_data[['Facility','penta3_ssum']][Isiolo_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Isiolo_data[['Facility','sbd_ssum']][Isiolo_data['sbd_ssum']>20]
		st.write(freq_dist)

	Isiolo_skhis = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Isiolo_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Isiolo_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Mombasa_County():
	st.title("Isiolo County")
	st.write("Below are Facilities covered")
	Isiolo_data = data[data['County'] == 'Isiolo County']
	Isiolo_df6 = df6[df6['County'] == 'Isiolo County']
	Isiolo_dfb6 = dfb6[dfb6['County'] == 'Isiolo County']
	Isiolo_dfc6 = dfc6[dfc6['County'] == 'Isiolo County']
	Isiolo_dfd6 = dfd6[dfd6['County'] == 'Isiolo County']
	st.write(Isiolo_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Isiolo_data['lat']),
		'longitude': min(Isiolo_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Isiolo_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Isiolo_ssum = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Isiolo_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Isiolo_fp_ssum = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Isiolo_fp_skhis = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Isiolo_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Isiolo_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Isiolo_data[['Facility','anc_ssum']][Isiolo_data['anc_ssum'] >=-50]
		freq_dist1 = Isiolo_data[['Facility','anc_ssum']][Isiolo_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Isiolo_data[['Facility','penta3_ssum']][Isiolo_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Isiolo_data[['Facility','sbd_ssum']][Isiolo_data['sbd_ssum']>20]
		st.write(freq_dist)

	Isiolo_skhis = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Isiolo_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Isiolo_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Muranga_County():
	st.title("Isiolo County")
	st.write("Below are Facilities covered")
	Isiolo_data = data[data['County'] == 'Isiolo County']
	Isiolo_df6 = df6[df6['County'] == 'Isiolo County']
	Isiolo_dfb6 = dfb6[dfb6['County'] == 'Isiolo County']
	Isiolo_dfc6 = dfc6[dfc6['County'] == 'Isiolo County']
	Isiolo_dfd6 = dfd6[dfd6['County'] == 'Isiolo County']
	st.write(Isiolo_data)
	
	st.deck_gl_chart(
		viewport={
		'latitude': min(Isiolo_data['lat']),
		'longitude': min(Isiolo_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Isiolo_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	Isiolo_ssum = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	#st.write(Baringo_ssum.head())
	subset_ssum = pd.melt(Isiolo_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

	Isiolo_fp_ssum = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_ssum','COC_ssum','E Pills_ssum',
	'Injection_ssum','IUCD_ssum','Implant_ssum','BTL_ssum']]
	Isiolo_fp_skhis = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','POP_skhis','COC_skhis','E Pills_skhis',
	'Injection_skhis','IUCD_skhis','Implant_skhis','BTL_skhis']]
	#st.write(Baringo_ssum.head())
	fp_ssum = pd.melt(Isiolo_fp_ssum, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	fp_skhis = pd.melt(Isiolo_fp_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='FP Commodities',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_ssum)
	fpchart = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_ssum, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchart.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchart)

	indica = ["Select an indicator","ANC","Penta3","SBA"]
	choice = st.sidebar.selectbox("Source and Summary",indica)
	if choice == "ANC":
		freq_dist = Isiolo_data[['Facility','anc_ssum']][Isiolo_data['anc_ssum'] >=-50]
		freq_dist1 = Isiolo_data[['Facility','anc_ssum']][Isiolo_data['anc_ssum']>=-100]
		fdb1 = (pd.concat([freq_dist,freq_dist1], axis=1))
		st.write(fdb1)
	elif choice == "Penta3":
		freq_dist = Isiolo_data[['Facility','penta3_ssum']][Isiolo_data['penta3_ssum']>20]
		st.write(freq_dist)
	elif choice == "SBA":
		freq_dist = Isiolo_data[['Facility','sbd_ssum']][Isiolo_data['sbd_ssum']>20]
		st.write(freq_dist)

	Isiolo_skhis = Isiolo_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_skhis','penta3_skhis','sbd_skhis']]
	#st.write(Baringo_skhis.head())
	subset_skhis = pd.melt(Isiolo_skhis, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	#st.write(subset_skhis)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
	fpchartb = sns.catplot(x="FP Commodities", y="Over/Under Reporting", data=fp_skhis, hue="Facility", height=4, aspect=1.5, kind="swarm")
	fpchartb.set_xticklabels(rotation=23, horizontalalignment = 'right')
	st.pyplot(fpchartb)
	#st.write(Baringo_df6)
	chart = sns.catplot(x="completeness", y="Availability", data=Isiolo_df6, hue="ssum", height=4, aspect=1.5, kind="bar")
	chart.set_xticklabels(rotation=15, horizontalalignment = 'right')
	st.pyplot(chart)

def Nairobi_County():
	st.title("Nairobi County")
	st.write("Below are Facilities covered")
	Nairobi_data = data[data['County'] == 'Nairobi County']
	st.write(Nairobi_data)

	st.deck_gl_chart(
		viewport={
		'latitude': min(Nairobi_data['lat']),
		'longitude': min(Nairobi_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Nairobi_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
         'auto_highlight': True
		}])

	indicator = ['anc_ssum','penta3_ssum','sbd_ssum']
	dataset = Nairobi_data.groupby('Sub-County')[indicator].sum()
	df= dataset.T
	Nairobi_df = Nairobi_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	st.write(Nairobi_df.head())
	subset = pd.melt(Nairobi_df, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	st.write(subset)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

def Nakuru_County():
	st.title("Nakuru County")
	st.write("Below are Facilities covered")
	Nakuru_data = data[data['County'] == 'Nakuru County']
	st.write(Nakuru_data)

	st.deck_gl_chart(
		viewport={
		'latitude': min(Nakuru_data['lat']),
		'longitude': min(Nakuru_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Nakuru_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
         'auto_highlight': True
		}])

	indicator = ['anc_ssum','penta3_ssum','sbd_ssum']
	dataset = Nakuru_data.groupby('Sub-County')[indicator].sum()
	df= dataset.T
	Nakuru_df = Nakuru_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	st.write(Nakuru_df.head())
	subset = pd.melt(Nakuru_df, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	st.write(subset)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

def Nandi_County():
	st.title("Nandi County")
	st.write("Below are Facilities covered")
	Nandi_data = data[data['County'] == 'Nandi County']
	st.write(Nandi_data)

	st.deck_gl_chart(
		viewport={
		'latitude': min(Nandi_data['lat']),
		'longitude': min(Nandi_data['lon']),
		'zoom': 9,
		'pitch': 100
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Nandi_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
        'auto_highlight': True
		}])

	indicator = ['anc_ssum','penta3_ssum','sbd_ssum']
	dataset = Nandi_data.groupby('Sub-County')[indicator].sum()
	df= dataset.T
	Nandi_df = Nandi_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	st.write(Nandi_df.head())
	subset = pd.melt(Nandi_df, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	st.write(subset)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

def Narok_County():
	st.title("Narok County")
	st.write("Below are Facilities covered")
	Narok_data = data[data['County'] == 'Narok County']
	st.write(Narok_data)

	st.deck_gl_chart(
		viewport={
		'latitude': min(Narok_data['lat']),
		'longitude': min(Narok_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Narok_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
         'auto_highlight': True
		}])

	indicator = ['anc_ssum','penta3_ssum','sbd_ssum']
	dataset = Narok_data.groupby('Sub-County')[indicator].sum()
	df= dataset.T
	Narok_df = Narok_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	st.write(Narok_df.head())
	subset = pd.melt(Narok_df, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	st.write(subset)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

def Nyamira_County():
	st.title("Nyamira County")
	st.write("Below are Facilities covered")
	Nyamira_data = data[data['County'] == 'Nyamira County']
	st.write(Nyamira_data)

	st.deck_gl_chart(
		viewport={
		'latitude': min(Nyamira_data['lat']),
		'longitude': min(Nyamira_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Nyamira_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
         'auto_highlight': True
		}])

	indicator = ['anc_ssum','penta3_ssum','sbd_ssum']
	dataset = Nyamira_data.groupby('Sub-County')[indicator].sum()
	df= dataset.T
	Nyamira_df = Nyamira_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	st.write(Nyamira_df.head())
	subset = pd.melt(Nyamira_df, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	st.write(subset)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

def Nyandarua_County():
	st.title("Nyandarua County")
	st.write("Below are Facilities covered")
	Nyandarua_data = data[data['County'] == 'Nyandarua County']
	st.write(Nyandarua_data)

	st.deck_gl_chart(
		viewport={
		'latitude': min(Nyandarua_data['lat']),
		'longitude': min(Nyandarua_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Nyandarua_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
         'auto_highlight': True
		}])

	indicator = ['anc_ssum','penta3_ssum','sbd_ssum']
	dataset = Nyandarua_data.groupby('Sub-County')[indicator].sum()
	df= dataset.T
	Nyandarua_df = Nyandarua_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	st.write(Nyandarua_df.head())
	subset = pd.melt(Nyandarua_df, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	st.write(subset)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

def Nyeri_County():
	st.title("Nyeri County")
	st.write("Below are Facilities covered")
	Nyeri_data = data[data['County'] == 'Nyeri County']
	st.write(Nyeri_data)

	st.deck_gl_chart(
		viewport={
		'latitude': min(Nyeri_data['lat']),
		'longitude': min(Nyeri_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Nyeri_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
         'auto_highlight': True
		}])

	indicator = ['anc_ssum','penta3_ssum','sbd_ssum']
	dataset = Nyeri_data.groupby('Sub-County')[indicator].sum()
	df= dataset.T
	Nyeri_df = Nyeri_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	st.write(Nyeri_df.head())
	subset = pd.melt(Nyeri_df, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	st.write(subset)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

def Samburu_County():
	st.title("Samburu County")
	st.write("Below are Facilities covered")
	Samburu_data = data[data['County'] == 'Samburu County']
	st.write(Samburu_data)

	st.deck_gl_chart(
		viewport={
		'latitude': min(Samburu_data['lat']),
		'longitude': min(Samburu_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Samburu_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
         'auto_highlight': True
		}])

	indicator = ['anc_ssum','penta3_ssum','sbd_ssum']
	dataset = Samburu_data.groupby('Sub-County')[indicator].sum()
	df= dataset.T
	Samburu_df = Samburu_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	st.write(Samburu_df.head())
	subset = pd.melt(Samburu_df, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	st.write(subset)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

def Siaya_County():
	st.title("Siaya County")
	st.write("Below are Facilities covered")
	Siaya_data = data[data['County'] == 'Siaya County']
	st.write(Siaya_data)

	st.deck_gl_chart(
		viewport={
		'latitude': min(Siaya_data['lat']),
		'longitude': min(Siaya_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Siaya_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
         'auto_highlight': True
		}])

	indicator = ['anc_ssum','penta3_ssum','sbd_ssum']
	dataset = Siaya_data.groupby('Sub-County')[indicator].sum()
	df= dataset.T
	Siaya_df = Siaya_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	st.write(Siaya_df.head())
	subset = pd.melt(Siaya_df, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	st.write(subset)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

def Taita_Taveta_County():
	st.title("Taita Taveta County")
	st.write("Below are Facilities covered")
	TaitaTaveta_data = data[data['County'] == 'Taita Taveta County']
	st.write(TaitaTaveta_data)

	st.deck_gl_chart(
		viewport={
		'latitude': min(TaitaTaveta_data['lat']),
		'longitude': min(TaitaTaveta_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': TaitaTaveta_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
         'auto_highlight': True
		}])

	indicator = ['anc_ssum','penta3_ssum','sbd_ssum']
	dataset = TaitaTaveta_data.groupby('Sub-County')[indicator].sum()
	df= dataset.T
	TaitaTaveta_df = TaitaTaveta_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	st.write(TaitaTaveta_df.head())
	subset = pd.melt(TaitaTaveta_df, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	st.write(subset)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

def Tana_River_County():
	st.title("Tana River County")
	st.write("Below are Facilities covered")
	TanaRiver_data = data[data['County'] == 'Tana River County']
	st.write(TanaRiver_data)

	st.deck_gl_chart(
		viewport={
		'latitude': min(TanaRiver_data['lat']),
		'longitude': min(TanaRiver_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': TanaRiver_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
         'auto_highlight': True
		}])

	indicator = ['anc_ssum','penta3_ssum','sbd_ssum']
	dataset = TanaRiver_data.groupby('Sub-County')[indicator].sum()
	df= dataset.T
	TanaRiver_df = TanaRiver_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	st.write(TanaRiver_df.head())
	subset = pd.melt(TanaRiver_df, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	st.write(subset)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

def Tharaka_Nithi_County():
	st.title("Tharaka Nithi County")
	st.write("Below are Facilities covered")
	TharakaNithi_data = data[data['County'] == 'Tharaka Nithi County']
	st.write(TharakaNithi_data)

	st.deck_gl_chart(
		viewport={
		'latitude': min(TharakaNithi_data['lat']),
		'longitude': min(TharakaNithi_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': TharakaNithi_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
         'auto_highlight': True
		}])

	indicator = ['anc_ssum','penta3_ssum','sbd_ssum']
	dataset = TharakaNithi_data.groupby('Sub-County')[indicator].sum()
	df= dataset.T
	TharakaNithi_df = TharakaNithi_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	st.write(TharakaNithi_df.head())
	subset = pd.melt(TharakaNithi_df, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	st.write(subset)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

def Trans_Nzoia_County():
	st.title("Trans-Nzoia County")
	st.write("Below are Facilities covered")
	TransNzoia_data = data[data['County'] == 'Trans-Nzoia County']
	st.write(TransNzoia_data)

	st.deck_gl_chart(
		viewport={
		'latitude': min(TransNzoia_data['lat']),
		'longitude': min(TransNzoia_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': TransNzoia_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
         'auto_highlight': True
		}])

	indicator = ['anc_ssum','penta3_ssum','sbd_ssum']
	dataset = TransNzoia_data.groupby('Sub-County')[indicator].sum()
	df= dataset.T
	TransNzoia_df = TransNzoia_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	st.write(TransNzoia_df.head())
	subset = pd.melt(TransNzoia_df, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	st.write(subset)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

def Turkana_County():
	st.title("Turkana County")
	st.write("Below are Facilities covered")
	Turkana_data = data[data['County'] == 'Turkana County']
	st.write(Turkana_data)

	st.deck_gl_chart(
		viewport={
		'latitude': min(Turkana_data['lat']),
		'longitude': min(Turkana_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Turkana_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
         'auto_highlight': True
		}])

	indicator = ['anc_ssum','penta3_ssum','sbd_ssum']
	dataset = Turkana_data.groupby('Sub-County')[indicator].sum()
	df= dataset.T
	Turkana_df = Turkana_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	st.write(Turkana_df.head())
	subset = pd.melt(Turkana_df, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	st.write(subset)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

def Uasin_Gishu_County():
	st.title("Uasin Gishu County")
	st.write("Below are Facilities covered")
	UasinGishu_data = data[data['County'] == 'Uasin Gishu County']
	st.write(UasinGishu_data)

	st.deck_gl_chart(
		viewport={
		'latitude': min(UasinGishu_data['lat']),
		'longitude': min(UasinGishu_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': UasinGishu_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
         'auto_highlight': True
		}])

	indicator = ['anc_ssum','penta3_ssum','sbd_ssum']
	dataset = UasinGishu_data.groupby('Sub-County')[indicator].sum()
	df= dataset.T
	UasinGishu_df = UasinGishu_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	st.write(UasinGishu_df.head())
	subset = pd.melt(UasinGishu_df, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	st.write(subset)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

def Vihiga_County():
	st.title("Vihiga County")
	st.write("Below are Facilities covered")
	Vihiga_data = data[data['County'] == 'Vihiga County']
	st.write(Vihiga_data)

	st.deck_gl_chart(
		viewport={
		'latitude': min(Vihiga_data['lat']),
		'longitude': min(Vihiga_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Vihiga_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
         'auto_highlight': True
		}])

	indicator = ['anc_ssum','penta3_ssum','sbd_ssum']
	dataset = Vihiga_data.groupby('Sub-County')[indicator].sum()
	df= dataset.T
	Vihiga_df = Vihiga_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	st.write(Vihiga_df.head())
	subset = pd.melt(Vihiga_df, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	st.write(subset)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

def Wajir_County():
	st.title("Wajir County")
	st.write("Below are Facilities covered")
	Wajir_data = data[data['County'] == 'Wajir County']
	st.write(Wajir_data)

	st.deck_gl_chart(
		viewport={
		'latitude': min(Wajir_data['lat']),
		'longitude': min(Wajir_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': Wajir_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
         'auto_highlight': True
		}])

	indicator = ['anc_ssum','penta3_ssum','sbd_ssum']
	dataset = Wajir_data.groupby('Sub-County')[indicator].sum()
	df= dataset.T
	Wajir_df = Wajir_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	st.write(Wajir_df.head())
	subset = pd.melt(Wajir_df, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	st.write(subset)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()

def West_Pokot_County():
	st.title("West Pokot County")
	st.write("Below are Facilities covered")
	WestPokot_data = data[data['County'] == 'West Pokot County']
	st.write(WestPokot_data)

	st.deck_gl_chart(
		viewport={
		'latitude': min(WestPokot_data['lat']),
		'longitude': min(WestPokot_data['lon']),
		'zoom': 9,
		'pitch': 50
		},
		layers=[{
		'type': 'ScatterplotLayer',
		'data': WestPokot_data,
		'getRadius': 1000,
		'getFillColor':[225,25,0],
		'pickable': True,
         'auto_highlight': True
		}])

	indicator = ['anc_ssum','penta3_ssum','sbd_ssum']
	dataset = WestPokot_data.groupby('Sub-County')[indicator].sum()
	df= dataset.T
	WestPokot_df = WestPokot_data[['County','Sub-County','Facility','MFLCODE','mfl','anc_ssum','penta3_ssum','sbd_ssum']]
	st.write(WestPokot_df.head())
	subset = pd.melt(WestPokot_df, 
		id_vars=['County','Sub-County','Facility','MFLCODE','mfl'],
		var_name='indicator',
		value_name='Over/Under Reporting'
		)
	st.write(subset)
	sns.catplot(x="indicator", y="Over/Under Reporting", data=subset, hue="Facility", height=4, aspect=1.5, kind="swarm")
	st.pyplot()
# Main Function
def main():
	pages = OrderedDict([("Baringo County", Baringo_County), ("Bomet County", Bomet_County), ("Bungoma County", Bungoma_County), 
	("Busia County", Busia_County), ("Elgeyo-Marakwet County", Elgeyo_Marakwet_County), ("Embu County", Embu_County),
	("Garissa County", Garissa_County), ("Homa Bay County", Homa_Bay_County), ("Isiolo County", Isiolo_County),
	("Kajiado County", Kajiado_County), ("Kakamega County", Kakamega_County), ("Kericho County", Kericho_County),
	("Kiambu County", Kiambu_County), ("Kilifi County", Kilifi_County), ("Kirinyaga County", Kirinyaga_County),
	("Kisii County", Kisii_County), ("Kisumu County", Kisumu_County), ("Kitui County", Kitui_County),
	("Kwale County", Kwale_County), ("Laikipia County", Laikipia_County), ("Lamu County", Lamu_County), ("Machakos County", Machakos_County),
	("Makueni County", Makueni_County), ("Mandera County", Mandera_County), ("Marsabit County", Marsabit_County), ("Meru County", Meru_County), 
	("Migori County", Migori_County), ("Mombasa County", Mombasa_County), ("Muranga County", Muranga_County),
	("Nairobi County", Nairobi_County), ("Nakuru County", Nakuru_County), ("Nandi County", Nandi_County), ("Narok County", Narok_County),
	("Nyamira County", Nyamira_County), ("Nyandarua County", Nyandarua_County),
	("Nyeri County", Nyeri_County), ("Samburu County", Samburu_County), ("Siaya County", Siaya_County), ("Taita Taveta County", Taita_Taveta_County),
	("Tana River County", Tana_River_County), ("Tharaka Nithi County", Tharaka_Nithi_County), ("Trans-Nzoia County", Trans_Nzoia_County),
	("Turkana County", Turkana_County), ("Uasin Gishu County", Uasin_Gishu_County), ("Vihiga County", Vihiga_County), ("Wajir County", Wajir_County),
	("West Pokot County", West_Pokot_County)])

	st.sidebar.title("CDV 2020")
	page = st.sidebar.selectbox("Select County.", list(pages.keys()))

	pages[page]()


if __name__ == '__main__':
	main()