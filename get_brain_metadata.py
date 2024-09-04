#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 14:25:31 2024

@author: user
"""
import pandas as pd
import numpy as np

def get_metadata(mouse_num='all'):

    metadata = [
        #%%% P60 brains
        # P60   ### okay overall --- just bad VisPL registration
                
            ### MISSING A BLOCK FROM LEFT HEMISPHERE CORTEX --- likely due to crash during analysis
        
        {'path':'/media/user/8TB_HDD/20231216_M127_MoE_P56_delip_RIMS_RI_1489_sunflower_80perc/',
          'name': 'M127_P60_fused',
          'exp':'P60', 'num':'M127',
          'sex':'M',   'age':60, 'thresh':500, 'weight':20,'Exact age': 62,

           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          #   ### ANTS DONE
          
          # ### use left hemisphere for VISUAL/AUD --- didnt make much difference
          # 'exclude':[
          #     ['VIS',  'L'],
          #     ['AUD', 'L'],
          #     ]
            'exclude':[
                ['VS',  'B'],  ### ventricles
                
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

                
                
                ]
          },   
                 
        #   ### P60 --- outlier areas... also harder to register due to very dim --- fix dimness!!!
        ### HIGHER THAN OTHER BRAINS --- maybe due to registration issues
        # {'path':'/media/user/20TB_HDD_NAS/20231117_M115_MoE_P56_5days_delip_RIMS_RI_1487_7days_5x_80perc_laser_REDO_WITH_SUNFLOWER/',
        #   'name': 'M115_MoE_P56_fused',
        #   'exp':'P60_NEW',   'num':'M115',
        #   'sex':'M',     'age':60, 'thresh':500,
          
        #   'pkl_to_use': 'MYELIN',    
        #   },              
        
        # ## EXCLUDE 229 - translation only stitching - maybe remove... --- a little high in auditory/visual, likey due to shifts in registration
        ### MAYBE CAN REMOVE NOW FULLY SINCE WE HAVE MORE FEMALES?
        {'path':'/media/user/8TB_HDD/20231031_M229_MoE_PVCre_P56_SHIELD_delp_RIMS_50perc_then_100perc_expanded_slightly_more_5x/',
          'name': 'M229_P56_noczi_fused',
            'exp':'P60', 'num':'M229',
            'sex':'F',     'age':60, 'thresh':450, 'weight':20,'Exact age': 58,
            
            'pkl_to_use': 'MYELIN',          
            # 'pkl_to_use': 'CUBIC',
            #   ### ANTS DONE
            
            ################# RIGHT HEMISPHERE HAS CEREBELLAR PIECE WHICH IS COUNTED RN IN ENTORHINAL CORTEX --- to clean manually? Or remove
                    ### or maybe just use left hemisphere only!!!
            'exclude':[
                ['VS',  'B'],
                
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

                ]
            },   
        
        
        # ### Good --- some registration issue in ILA/ORB/FRP
        {'path':'/media/user/20TB_HDD_NAS/20240103_M126_MoE_Reuse_delip_RIMS_RI_14926_sunflow_80perc/',
          'name': 'M126_MoE_P60_fused',
          'exp':'P60',   'num':'M126',
          'sex':'M',     'age':60, 'thresh':500, 'weight':19, 'Exact age': 62,
          
           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          #   ### ANTS DONE # ANTS - a little shaved off of the cortex near cerebellum --- exclude ECT???
                      ### so might want to use left hemisphere!!! Since right has layer 2 projected into layer 1 atm
                      
          ### use left hemisphere for ACA/ORB/ILA --- might not do that much...
            'exclude':[
                ['PL',  'L'],
                ['ORB', 'L'],
                ['ILA', 'L'],
                ['FRP', 'L'],
                ['ACA', 'L'],
                
                ### EXCLUDE ECT AS WELL...
                ['VS',  'B'],
                
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

                
                
                ]  
              
          },   
        
        ### All P56  ### EXCLUDE due to no sunflower oil
        # {'path':'/media/user/20TB_HDD_NAS/20231012_M230_MoE_PVCre_SHIELD_delip_RIMS_RI_1500_3days_5x/',
        #  'name': 'M230_P56_fused',     
        #   'exp':'P60_nosunflow',  'num':'M230',
        #   'sex':'F',              'age':60, 'thresh':400,
        #   },             
        
        
        
        
          # P60 CUBIC --- a bit more outlier-y (a bit higher in some areas in WM ect...) - was earlier done brain
        {'path':'/media/user/8TB_HDD/20231012_M223_MoE_Ai9_SHIELD_CUBIC_RIMS_RI_1500_3days_5x/',
          'name': 'M223_CUBIC_fused',   
          'exp':'P60',  'num':'M223',
          'sex':'M',    'age':60, 'thresh':400, 'weight':25.2,'Exact age': 59,
          
           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          
            'exclude':[
                ['VS',  'B'],
                
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

                ]
          }, 
        
        
        
        
        
        # ### P60 CUBIC new
        {'path':'/media/user/20TB_HDD_NAS/20240124_M299_MoE_P60_SHIELD_CUBIC7d_RIMS_RI_1489_sunflow_laser_80perc/',
          'name': 'M299_MoE_P60_CUBIC_fused',   
          'exp':'P60',  'num':'M299',
          'sex':'M',    'age':60, 'thresh':500, 'weight':25.5, 'Exact age': 58,
          
           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          #   ### ANTS DONE
          
            'exclude':[
                ['VS',  'B'],
                
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

                ]
          },   
        
        
        
        
        
            ### P60 CUBIC new with AAVs --- ANTS - shaved off a bit as well
          {
            'path':'/media/user/20TB_HDD_NAS/20240210_M254_MoE_P60_low_AAVs_SHIELD_CUBIC_7d_RIMS_2d_RI_1493_sunflower/',
              # 'path':'/media/user/4TB_SSD/20240210_M254_MoE_P60_low_AAVs_SHIELD_CUBIC_7d_RIMS_2d_RI_1493_sunflower/',
            'name': 'M254_MoE_P60_low_AAVs_fused',       
            'exp':'P60',   'num':'M254', 
            'sex':'F',     'age':60, 'thresh':350,  'weight':16, 'Exact age': 58,
            
            'pkl_to_use': 'MYELIN',          
            # 'pkl_to_use': 'CUBIC',
            #   ### ANTS DONE  --- right hemisphere ring WM near ECT is a bit off, not awful, but could just use left hemisphere
            
            'exclude':[
                ['VS',  'B'],
                
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

                ]
            },   
           
            ## P60 with HIGH AAVs - 3 channels!!!
        {'path':'/media/user/20TB_HDD_NAS/20240210_M256_MoE_P60_high_AAVs_SHIELD_CUBIC_7d_RIMS_2d_RI_1493_sunflow_60perc/',
          'name': 'M256_P60_MoE_high_AAVs_3chs_fused',            
          'exp':'P60',  'num':'M256',
          'sex':'F',     'age':60, 'thresh':350,  'weight':17.5, 'Exact age': 58,
          
           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          #   ### ANTS DONE
          
            'exclude':[
                ['VS',  'B'],
                
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

                ]
          },                            
                    
        
        
        ### P60   --- has some lower areas... no idea why. Not awful, but noticeable... OH --- it is a P56, not P60... maybe that's why?
        {'path':'/media/user/20TB_HDD_NAS_2/20240425_M260_MoE_P56_SHIELD_CUBIC_RIMS_RI_1493_fresh_100perc_488_30perc_638_80msec/',
          'name': 'M260_MoE_P60_fused',          
          'exp':'P60',  'num':'M260',
          'sex':'F',     'age':60, 'thresh':500, 'weight':18, 'Exact age': 56,
          
           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          #   ### ANTS DONE
          
            'exclude':[
                ['VS',  'B'],
                
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

                ]
          },           

                                    ### COPY
                                    # {'path':'/media/user/20TB_HDD_NAS_2/20240425_M260_MoE_P56_SHIELD_CUBIC_RIMS_RI_1493_fresh_100perc_488_30perc_638_80msec_COPY/',
                                    #   'name': 'M260_MoE_P60_fused',          
                                    #   'exp':'P60_NEW_NEW',  'num':'M260',
                                    #   'sex':'F',     'age':60, 'thresh':500,
                                    #   },       
                                    
        
        
        #%%% P240
        # 8 mos - tear in right ACA --- ALSO WAS done with Delip so really awful clearing deeper down
        {'path':'/media/user/20TB_HDD_NAS/20240125_M279_MoE_8mos_SHIELD_delip8d_RIMS_RI_1492_sunflow_laser_80perc/',
          'name': 'M279_MoE_8mos_fused',       
          'exp':'P240',  'num':'M279',
          'sex':'M',     'age':240, 'thresh':500, 'weight':34.5, 'Exact age': 242,
          
           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          # 'side':'L'
          #   ### ANTS DONE
            'exclude':[
                ['STR',  'B'],  ### Striatum --- this includes olfactory tubercule and deeper structures
                ['TH', 'B'],   ### Thalamus
                ['HY', 'B'],   ### Hypothalamus
                ['IG', 'B'],
                ### EXCLUDE ECT AS WELL...
                ['IB', 'B'],
                
                
                ['CB',  'B'],
                
                ['VS',  'B'],
                
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

                ]  
          
          
          },   
        
        # 8 mos --- ANTS
        {'path':'/media/user/20TB_HDD_NAS/20240210_M286_MoE_8mos_SHIELD_CUBIC_7d_RIMS_2d_RI_1493_sunflow_80perc/',
          'name': 'M286_8mos_MoE_fused',   
          'exp':'P240',  'num':'M286',
          'sex':'F',     'age':240, 'thresh':400,  'weight':29, 'Exact age': 243,
          
           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          #   ### ANTS DONE  - a little bit of cerebellum is segmented as cortex
            'exclude':[
                ['VS',  'B'],
                
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

                ]
          },   

                                                # 8 mos --- COPY
                                                # {'path':'/media/user/20TB_HDD_NAS/20240210_M286_MoE_8mos_SHIELD_CUBIC_7d_RIMS_2d_RI_1493_sunflow_80perc_COPY/',
                                                #   'name': 'M286_8mos_MoE_fused',   
                                                #   'exp':'P240_NEW_NEW',  'num':'M286',
                                                #   'sex':'F',     'age':240, 'thresh':400,
                                                #   },   
                                                    
            
        # ### 8 months - needs more registration for auditory + Vispl + frontal ORB + FRP --- PROBLEM CHILD
        {'path':'/media/user/20TB_HDD_NAS/20240228_M281_MoE_8mos_SHIELD_CUBIC_6d_RIMS_1d_RI_1493_reuse_sunflow_561_laser_AUTOFLUOR/',
          'name': 'M281_MoE_8mos_fused',       
          'exp':'P240',  'num':'M281',
          'sex':'M',     'age':240, 'thresh':400, 'weight':33, 'Exact age': 242,
          
           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          #   ### ANTS DONE
          
          ### Exclude - right hemisphere IC
          ### also overall needs more registration for auditory + Vispl + frontal ORB + FRP --- PROBLEM CHILD
            'exclude':[
                ['VS',  'B'],
                
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

                ]
          
          },                            
        
        ### 8 months  ### hole in top right cortex --- LOTS OF BLOOD
        {'path':'/media/user/20TB_HDD_NAS/20240227_M285_MoE_8months_SHIELD_CUBIC_7d_RIMS_RI_1493_sunflow/',
          'name': 'M285_MoE_8mos_fused',       
          'exp':'P240',  'num':'M285',
          'sex':'M',     'age':240, 'thresh':400, 'weight':25.5, 'Exact age': 243,
          
           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          ### ANTS DONE (looks good) --- yes brainreg
          
              ### High in FRP and VISpl --- but otherwise good
            'exclude':[
                ['CB',  'B'],
                ['VS',  'B'],
                
                
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

                ]
          
          },                            
                          


 
        
        #%%% P620
        
          #  22 mos --- USING NEW NORM MASKRCNN
        {'path':'/media/user/20TB_HDD_NAS/20240127_M271_MoE_OLD_22mos_SHIELD_CUBIC10d_RIMS_RI_1493_sunflow_laser_80perc/',
          'name': 'M271_MoE_OLD_22mos_fused',          
          'exp':'P620',  'num':'M271',
          'sex':'M',     'age':620, 'thresh':750, 'weight':26, 'Exact age': 616,
          
           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          #   ### ANTS DONE --- left hemisphere slightly better
            'exclude':[
                ['VS',  'B'],
                
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

                ]
          },             

                                                                                    #   #  22 mos --- with ANTS - copied
                                                                                    # {'path':'/media/user/20TB_HDD_NAS/20240127_M271_MoE_OLD_22mos_SHIELD_CUBIC10d_RIMS_RI_1493_sunflow_laser_80perc_COPY/',
                                                                                    #   'name': 'M271_MoE_OLD_22mos_fused',          
                                                                                    #   'exp':'P620_NEW_NEW',  'num':'M271',
                                                                                    #   'sex':'M',     'age':620, 'thresh':500,
                                                                                    #   },     
                                                                                    


        
        # ## OLD REDO M91 - switched back to one without ICP
        {'path':'/media/user/8TB_HDD/20240101_M91_OLD_MoE_delip_13d_CUBIC_7d_RIMS_RI_1489_sunflow_80perc_NO_FUSION/',
          'name': 'M91_OLD_CUBIC_fused',   
          'exp':'P620',  'num':'M91',
          'sex':'M',     'age':620, 'thresh':450, 'weight':'?', 'Exact age': 662,
          
           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          #   ### ANTS DONE --- left hemisphere better
          
          ### Torn areas to remove (or to replace with a side) --- LOTS MORE IN CORTEX AS WELL
          'exclude':[
              ['PL',  'B'],
              ['ORB', 'B'],
              ['ILA', 'B'],
              ['FRP', 'B'],
              ['ACA', 'B'],
              ['CB', 'B'],
              
              ['VS',  'B'],
              
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

              ]
          },         
        
                                                                            ### # OLD REDO M91 - RE-FUSED with ICP  - actually not great... shifted some regions more than others
                                                                            ### {'path':'/media/user/20TB_HDD_NAS/20240101_M91_OLD_MoE_delip_13d_CUBIC_7d_RIMS_RI_1489_sunflow_80perc_NO_FUSION/',
                                                                            ###   'name': 'M91_22mos_RE-FUSE_fused',         
                                                                            ###   'exp':'P620',  'num':'M91',
                                                                            ###   'sex':'M',     'age':620, 'thresh':400,
                                                                            ###   },                         

        ### OLD M97 22mos  ### ALSO VERY LOW CELL COUNT?
        {'path':'/media/user/20TB_HDD_NAS_2/20240308_M97_MoE_SHIELD_CUBIC_RIMS_RI_14925_14d_sunflow/',
          'name': 'M97_22mos_MoE_fused',         
          'exp':'P620',  'num':'M97',
          'sex':'M',     'age':620, 'thresh':400, 'weight':'?', 'Exact age': 624,
          
           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          ### ANTS DONE
    
            ### NO RIGHT OLFACTORY BULB CAUSING BIG ARTIFACTS
            
            ### Registering with CUBIC autofluor helped a little... especially if only use left side counts 
            'exclude':[
                ['all',  'L'],
                
                ['VS',  'B'],
                
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

                ]
          },           
        
        
        
        # # Excellent, new brain, no blood ---  ### *** ACA registration is off - PROBLEM CHILD (rest is okay though)
        {'path':'/media/user/4TB_SSD/20240501_M334_MoE_22mos_SHIELD_CUBIC_14d_RIMS_RI_1493_100p488_20p638_100msec/',
          'name': 'M334_MoE_22mos_fused',         
          'exp':'P620',  'num':'M334',
          'sex':'M',     'age':620, 'thresh':500,  'weight':'CHECK IN BOOK', 'Exact age':'CHECK IN BOOK',
          
           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          #   ### ANTS DONE
          ### --- NEITHER SIDE WAS GREAT FOR ECT AREA... MAYBE TOO MUCH MYELIN DIVISION??? TRY INCREASING THRESHOLD TO 450/500?
          
          ### *** ACA registration is off --- olfactory bulb is bulging back inwards too much... might need a segmentation tool to get the main brain... excluding olfactory and cerebellum...
            'exclude':[
                ['CB',  'L'],
                
                ['VS',  'B'],
                
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

                ]
          },                
 
                     
        # # 20 mos --- WEIRD - was lower cell count --- due to tissue tearing???
        #       {'path':'/media/user/8TB_HDD/20240109_M27_MoE_OLD_SHIELD_CUBIC_RI_RIMS_14926_sunflow/',
        #        'name': 'M27_MoE_22mos_fused',                     
        #         'exp':'20mos',   'num':'M27',
        #         'sex':'M',       'age':62, 'thresh':400,
        #         },   
        
        
        
        #%%
        # #%%% P800  ### SHOULD ACTUALLY BE P850!!! --- ### REALLY BAD - NO OLFACTORY BULBS AT ALL CAUSING HUGE ARTIFACTS
        ### 2.5 years OLD --- WAY TOO MUCH VASCULATURE --- REDONE WITH VESSEL SUB + NEWLY TRAINED NETWORK (NORM + NO REPLACE)
        {'path':'/media/user/20TB_HDD_NAS_2/20240214_1Otx7_Jaime-Tx_2yo_MoE_OLD_SHIELD_CUBIC_12d_RIMS_RI_1493_sunflow/',
          'name': '1Otx7_2yo_REDO_fused',       
          'exp':'P800',  'num':'1Otx7',
          'sex':'?',      'age':850, 'thresh':500,  'weight':'CHECK IN BOOK', 'Exact age':'CHECK IN BOOK',
          
           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          ### ANTS DONE --- left hemisphere much better than right hemisphere
          
          ### Torn areas to remove (or to replace with a side)
          
          ### REALLY BAD - NO OLFACTORY BULBS AT ALL CAUSING HUGE ARTIFACTS
          'exclude':[
              ['PL',  'B'],
              ['ORB', 'B'],
              ['ILA', 'B'],
              ['FRP', 'B'],
              ['ACA', 'B'],
              
              ['MB',  'B'],  ### added for now...
              ['IB', 'B'],
              ['CB', 'B'],
              
              ['VS',  'B'],
              
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

              ]
              
              
          }, 
                
        
        # ## OLD - needs vessel subtraction!!! --- poor registration in ECT/AUD/VISpl --- PROBLEM CHILD
        {'path':'/media/user/20TB_HDD_NAS_2/20240308_otx18_MoE_2yo_SHIELD_CUBIC_RIMS_RI_14925_14d_sunflow/',
          'name': 'otx18_MoE_2yo_fused',        
          'exp':'P800',  'num':'Otx18',
          'sex':'?',     'age':850, 'thresh':300, 'weight':'CHECK IN BOOK', 'Exact age':'CHECK IN BOOK',  ### actual age P800
          
           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          #   ### ANTS DONE
          ### --- also not great both sides ECT AREA... MAYBE TOO MUCH MYELIN DIVISION??? TRY INCREASING THRESHOLD TO 350/400?
          'exclude':[
              ['MB',  'B'],  ### added for now...
              ['IB', 'B'],
              ['CB', 'B'],
              
              ['VS',  'B'],
              
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

              ]
          
          
          },                            

                
                        # ## OLD - needs vessel subtraction!!! --- coming out a little low?... COPY
                        # {'path':'/media/user/20TB_HDD_NAS_2/20240308_otx18_MoE_2yo_SHIELD_CUBIC_RIMS_RI_14925_14d_sunflow_COPY/',
                        #   'name': 'otx18_MoE_2yo_fused',        
                        #   'exp':'P800_NEW_NEW',  'num':'Otx18',
                        #   'sex':'?',     'age':850, 'thresh':300,   ### actual age P800
                        #   #   ### ANTS DONE
                        #   ### --- also not great both sides ECT AREA... MAYBE TOO MUCH MYELIN DIVISION??? TRY INCREASING THRESHOLD TO 350/400?
                        #   },    
                        
        
             
        ### 2.5 years OLD not too bad vessel bleed through --- likely stil best to do vessel dampening
        {'path':'/media/user/20TB_HDD_NAS_2/20240307_5otx5_MoE_2yo_SHIELD_CUBIC_RIMS_RI_14925_14d_sunflow/',
          'name': '5otx5_2yo_fused',       
          'exp':'P800',  'num':'5Otx5',
          'sex':'?',      'age':850, 'thresh':300,  'weight':'CHECK IN BOOK', 'Exact age':'CHECK IN BOOK',
          
           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          ### ANTS DONE --- left hemisphere better
             'exclude':[
                 ['VS',  'B'],
                 
                 # Hypothalamic midline regions
                 ['PVZ',  'B'],
                 ['PVR',  'B'],
                 ['MBO',  'B'],                
                 # cortical subplate?
                 ['BMA',  'B'],
                 ['PA',  'B'],
                 # fiber tracts
                 ['cm', 'B'],  ### includes von, IIn, onl
                 ['pyd',  'B'],
                 ['cbp', 'B'],   ### cerebellar peduncles
                 




                 ]
          },         
        
        
        
        
        
        #%%% P950  --- OLDEST
        ## 2.5 years OLD --- better, still some vessels so need vessel sub
        {'path':'/media/user/20TB_HDD_NAS_2/20240419_Otx6_MoE_34mos_SHIELD_CUBIC_14d_RIMS_RI_14968_100perc_488_30perc_638_100msec_6x8_tiles/',
          'name': 'Otx6_34mos_MoE_fused',       
          'exp':'P800',  'num':'Otx6',
          'sex':'?',      'age':850, 'thresh':500,  'weight':'CHECK IN BOOK', 'Exact age':'CHECK IN BOOK',
          
           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          ### ANTS DONE --- brainreg done too
             'exclude':[
                 ['VS',  'B'],
                 
                 # Hypothalamic midline regions
                 ['PVZ',  'B'],
                 ['PVR',  'B'],
                 ['MBO',  'B'],                
                 # cortical subplate?
                 ['BMA',  'B'],
                 ['PA',  'B'],
                 # fiber tracts
                 ['cm', 'B'],  ### includes von, IIn, onl
                 ['pyd',  'B'],
                 ['cbp', 'B'],   ### cerebellar peduncles
                 


                 ]
          }, 
        
        
        
        
        #%%% FVB and CD1 comparisons
          # # FVB
        {'path':'/media/user/8TB_HDD/20231114_M147_MoE_FVB_7days_delip_RIMS_RI_1487_5days_5x_60perc_laser_SUNFLOWER/',
          'name': 'M147_FVB_fused',           
          'exp':'FVB',  'num':'M147',
          'sex':'F',    'age':58, 'thresh':450, 'weight':18, 'Exact age':58,   
          
           'pkl_to_use': 'MYELIN',          
          # 'pkl_to_use': 'CUBIC',
          ### ANTS DONE --- yes brainreg
          
          ### Ripped right hemisphere RSP
           'exclude':[
               ['RSP',  'L'],
               ['VS',  'B'],
               
               ['HY',  'B'],  ### poor clearing in deep layers due to delip buffer
               ['MED',  'B'],  ### poor clearing in deep layers due to delip buffer
               ['Vn',  'B'],  ### poor clearing in deep layers due to delip buffers
               
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

               ]

          },
         
          # ## FVB - SUPER BLOODY
          #  {'path':'/media/user/8TB_HDD/20231222_M154_FVB_delip_reused_RIMS_weird_no_sunflower1st_time_RIMS_RI_1489_sunflower_80perc/',
          #   'name': 'M154_FVB_fused',           
          #  'exp':'FVB',  'num':'M154',
          #  'sex':'M',    'age':59, 'thresh':500,
          
          #  },   
          
          ## FVB
          {'path':'/media/user/20TB_HDD_NAS/20240124_M155_FVB_MoE_SHIELD_delip7d_RIMS_PB_wash_CUBIC_2d_RIMS_RI_1489_sunflow_laser_60-60perc/',
            'name': 'M155_FVB_fused', 
            'exp':'FVB',  'num':'M155',
            'sex':'M',     'age':60, 'thresh':400, 'weight':26, 'Exact age':59, 
            
            'pkl_to_use': 'MYELIN',          
           # 'pkl_to_use': 'CUBIC',
           
            ### ANTS DONE --- cortex in back lost as cerebellum for both hemispheres!!! --- yes brainreg
            ### Ripped right hemisphere RSP
           'exclude':[
               ['RSP',  'L'],
               
               ### ALSO frontal lobe --- ACA? PL? ORB?
               ['VS',  'B'],
               
               ['HY',  'B'],  ### poor clearing in deep layers due to delip buffer
               ['MED',  'B'],  ### poor clearing in deep layers due to delip buffer
               ['Vn',  'B'],  ### poor clearing in deep layers due to delip buffers
               
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

               ]
            },                            
                      
          ## FVB
          {'path':'/media/user/20TB_HDD_NAS/20240125_M152_FVB_MoE_SHIELD_delip_RIMS_RI_1492_sunflower_80perc/',
            'name': 'M152_FVB_fused',        
            'exp':'FVB',  'num':'M152',
            'sex':'M',     'age':60, 'thresh':500, 'weight':22.4, 'Exact age':59, 
            
            'pkl_to_use': 'MYELIN',          
           # 'pkl_to_use': 'CUBIC',
            ### ANTS DONE --- yes brainreg
            ### Ripped right hemisphere RSP
           'exclude':[
               ['RSP',  'L'],
               ['VS',  'B'],
               
               ['HY',  'B'],  ### poor clearing in deep layers due to delip buffer
               ['MED',  'B'],  ### poor clearing in deep layers due to delip buffer
               ['Vn',  'B'],  ### poor clearing in deep layers due to delip buffers
               
                # Hypothalamic midline regions
                ['PVZ',  'B'],
                ['PVR',  'B'],
                ['MBO',  'B'],                
                # cortical subplate?
                ['BMA',  'B'],
                ['PA',  'B'],
                # fiber tracts
                ['cm', 'B'],  ### includes von, IIn, onl
                ['pyd',  'B'],
                ['cbp', 'B'],   ### cerebellar peduncles
                

               ]
            },                            
                      
          # ## FVB - a bit bloody, redoing right now
           {'path':'/media/user/20TB_HDD_NAS/20240125_M146_FVB_MoE_SHIELD_delip8d_RIMS_RI_1492_sunflow_laser_80perc/',
             'name': 'M146_FVB_fused',  
             'exp':'FVB',  'num':'M146',
             'sex':'F',     'age':60, 'thresh':500, 'weight':18.1, 'Exact age':58, 
             ### ANTS DONE --- tiny bit of cortex clipped as cerebellum both hemispheres --- yes brainreg
             
             'pkl_to_use': 'MYELIN',      ### for now, name not changed  
             
             'exclude':[
                 ['VS',  'B'],
                 
                 ['HY',  'B'],  ### poor clearing in deep layers due to delip buffer
                 ['MED',  'B'],  ### poor clearing in deep layers due to delip buffer
                 ['Vn',  'B'],  ### poor clearing in deep layers due to delip buffers
                 
                 # Hypothalamic midline regions
                 ['PVZ',  'B'],
                 ['PVR',  'B'],
                 ['MBO',  'B'],                
                 # cortical subplate?
                 ['BMA',  'B'],
                 ['PA',  'B'],
                 # fiber tracts
                 ['cm', 'B'],  ### includes von, IIn, onl
                 ['pyd',  'B'],
                 ['cbp', 'B'],   ### cerebellar peduncles
                 

                 ]
             },            
          
        # # # CD1
           #  {'path':'/media/user/8TB_HDD/20231216_M170_MoE_CD1_delip_RIMS_RI_1489_sunflower_80perc/',
           #  'name': 'M170_CD1_fused',         
           #  'exp':'CD1',  'num':'M170',
           #  'sex':'M',    'age':62, 'thresh':500,
            
           #  'pkl_to_use': 'MYELIN',          
           # # 'pkl_to_use': 'CUBIC',
           #  ### ANTS DONE --- piece of cerebellum clipped as cortex both hemispheres --- yes brainreg
           #  ### Ripped right hemisphere RSP
           # 'exclude':[
           #     ['VS',  'B'],
               
           #     ['HY',  'B'],  ### poor clearing in deep layers due to delip buffer
           #     ['MED',  'B'],  ### poor clearing in deep layers due to delip buffer
           #     ['Vn',  'B'],  ### poor clearing in deep layers due to delip buffers
               
           #      # Hypothalamic midline regions
           #      ['PVZ',  'B'],
           #      ['PVR',  'B'],
           #      ['MBO',  'B'],                
           #      # cortical subplate?
           #      ['BMA',  'B'],
           #      ['PA',  'B'],
           #      # fiber tracts
           #      ['cm', 'B'],  ### includes von, IIn, onl
           #      ['pyd',  'B'],
           #      ['cbp', 'B'],   ### cerebellar peduncles
                

           #     ]
           #  }, 
    
        # ### CD1
        # {'path':'/media/user/20TB_HDD_NAS/20240116_M172_CD1_MoE_SHIELD_delip_RIMS_RI_1489_sunflow_laser_70-60perc/',
        #     'name': 'M172_CD1_fused',  
        #   'exp':'CD1',  'num':'M172',
        #   'sex':'F',     'age':121, 'thresh':500,
          
        #    'pkl_to_use': 'MYELIN',          
        #   # 'pkl_to_use': 'CUBIC',
        #   ### ANTS DONE --- piece of cerebellum clipped as cortex both hemispheres --- yes brainreg
        #     ### Ripped right hemisphere RSP
        #    'exclude':[
        #        ['RSP',  'L'],
        #        ['VS',  'B'],
               
        #        ['HY',  'B'],  ### poor clearing in deep layers due to delip buffer
        #        ['MED',  'B'],  ### poor clearing in deep layers due to delip buffers
        #        ['Vn',  'B'],  ### poor clearing in deep layers due to delip buffers
               
        #         # Hypothalamic midline regions
        #         ['PVZ',  'B'],
        #         ['PVR',  'B'],
        #         ['MBO',  'B'],                
        #         # cortical subplate?
        #         ['BMA',  'B'],
        #         ['PA',  'B'],
        #         # fiber tracts
        #         ['cm', 'B'],  ### includes von, IIn, onl
        #         ['pyd',  'B'],
        #         ['cbp', 'B'],   ### cerebellar peduncles
                


        #        ]
        #   },                            
          
        
          # # P60 CD1 - warped - NO LOCATION
             # {'path':'/media/user/20TB_HDD_NAS_2/20231116_M169_MoE_CD1_SHIELD_RIMS_RI1487_5x_60perc_laser_SUNFLOWER_WARPED/',
             #  'name': 'M169_CD1_fused',      
             #  'exp':'CD1_NEW',    'num':'M169',
             #  'sex':'M',          'age':63,'thresh':500,
              
             #   'pkl_to_use': 'MYELIN',          
              
              
             #  },     
                
                    
        ###########################################
        #%%% CUPRIZONE
          ## For CUPRIZONE
        {'path':'/media/user/20TB_HDD_NAS/20240227_M265_Cuprizone_6wks_SHIELD_CUBIC_7d_RIMS_RI_1493_reuse_sunflow/',
            'name': 'M265_Cuprizone_6wks_fused',  
            'exp':'Cuprizone',   'num':'M265',
            'sex':'M',           'age':100, 'thresh':0, 'weight':24, 'Exact age':101, 
            
            'pkl_to_use': 'MYELIN',      ### for now, name not changed    
           # 'pkl_to_use': 'CUBIC',
                ### ANTS - a bit problematic, parts of severed Cerebellum are being added to cortex
                'exclude':[
                    ['VS',  'B'],
                    
                    # Hypothalamic midline regions
                    ['PVZ',  'B'],
                    ['PVR',  'B'],
                    ['MBO',  'B'],                
                    # cortical subplate?
                    ['BMA',  'B'],
                    ['PA',  'B'],
                    # fiber tracts
                    ['cm', 'B'],  ### includes von, IIn, onl
                    ['pyd',  'B'],
                    ['cbp', 'B'],   ### cerebellar peduncles
                    


                    ]
            },
                                      
        ## For CUPRIZONE
        {'path':'/media/user/20TB_HDD_NAS/20240215_M266_MoE_CUPRIZONE_6wks_SHIELD_CUBIC_7d_RIMS_RI_1493_sunflow/',
          'name': 'M266_MoE_CUPRIZONE_6wks_fused',  
          'exp':'Cuprizone',  'num':'M266',
          'sex':'M',          'age':100, 'thresh':0, 'weight':22.5, 'Exact age':101,
          
            'pkl_to_use': 'MYELIN',      ### for now, name not changed    
           # 'pkl_to_use': 'CUBIC',
             'exclude':[
                 ['VS',  'B'],
                 
                 # Hypothalamic midline regions
                 ['PVZ',  'B'],
                 ['PVR',  'B'],
                 ['MBO',  'B'],                
                 # cortical subplate?
                 ['BMA',  'B'],
                 ['PA',  'B'],
                 # fiber tracts
                 ['cm', 'B'],  ### includes von, IIn, onl
                 ['pyd',  'B'],
                 ['cbp', 'B'],   ### cerebellar peduncles
                 


                 ]
          },
        
        
        ### CUPRIZONE --- quite bloody
        {'path':'/media/user/20TB_HDD_NAS_2/20240422_M267_REDO_REDO_MoE_CUP_6wks_SHIELD_CUBIC_RIMS_RI_1496_2d_after_PB_wash_100perc_488_60perc_638_100msec/',
          'name': 'M267_REDO_REDO_Cup6wks_fused',  
          'exp':'Cuprizone',  'num':'M267',
          'sex':'M',          'age':100, 'thresh':0, 'weight':22, 'Exact age':101,
          
            'pkl_to_use': 'MYELIN',      ### for now, name not changed    
           # 'pkl_to_use': 'CUBIC',
             'exclude':[
                 ['VS',  'B'],
                 
                 # Hypothalamic midline regions
                 ['PVZ',  'B'],
                 ['PVR',  'B'],
                 ['MBO',  'B'],                
                 # cortical subplate?
                 ['BMA',  'B'],
                 ['PA',  'B'],
                 # fiber tracts
                 ['cm', 'B'],  ### includes von, IIn, onl
                 ['pyd',  'B'],
                 ['cbp', 'B'],   ### cerebellar peduncles
                 


                 ]
          },        
        
        
        # # # # CUPRIZONE --- started cuprizone too early
        # # # {'path':'/media/user/8TB_HDD/20231115_M139_MoE_CasprtdT_Cuprizone_6wk__SHIELD_RIMS_RI1487_5x_60perc_laser_SUNFLOWER/',
        # # #     'name': 'M139_cuprizone_fused',       
        # # #   'exp':'Cuprizone',   'num':'M139',
        # # #   'sex':'M',           'age':120, 'thresh':0,
        # # #   },
         
        
        # # ###########################
          
        ## CUPRIZONE + RECOVERY 3wks
        {'path':'/media/user/20TB_HDD_NAS_2/20240419_M312_REDO_REDO_PB_washed_GOOD_SHIELD_CUBIC_7d_RI_RIMS_14968_100perc_488_50perc_638_100msec/',
            'name': 'M312_REDO_REDO_Cup_3wksRecov_fused',     
            'exp':'Recovery',   'num':'M312',
            'sex':'F',           'age':120, 'thresh':0, 'weight':18.4, 'Exact age':120,   #500,
            ### ANTS registered - but not brainreg --- yes brainreg
            
            'pkl_to_use': 'MYELIN',      ### for now, name not changed    
           # 'pkl_to_use': 'CUBIC',
             'exclude':[
                 ['VS',  'B'],
                 
                 # Hypothalamic midline regions
                 ['PVZ',  'B'],
                 ['PVR',  'B'],
                 ['MBO',  'B'],                
                 # cortical subplate?
                 ['BMA',  'B'],
                 ['PA',  'B'],
                 # fiber tracts
                 ['cm', 'B'],  ### includes von, IIn, onl
                 ['pyd',  'B'],
                 ['cbp', 'B'],   ### cerebellar peduncles
                 



                 ]
            }, 
                       
        
        ## CUPRIZONE + RECOVERY 3wks
        {'path':'/media/user/20TB_HDD_NAS_2/20240420_M310_REDO_REDO_PB_washed_MoE_CUP_6wks_RECOV_3wks_RI_RIMS_1496_2d_after_wash_100perc_488_60perc_638_100msec/',
            'name': 'M310_REDO_REDO_CUP6wks_REC3wks_fused',     
            'exp':'Recovery',   'num':'M310',
            'sex':'M',           'age':120, 'thresh':0,   'weight':20, 'Exact age':120,   #500
            ### ANTS registered - but not brainreg --- yes brainreg
            
            'pkl_to_use': 'MYELIN',      ### for now, name not changed    
           # 'pkl_to_use': 'CUBIC',
             'exclude':[
                 ['VS',  'B'],
                 
                 # Hypothalamic midline regions
                 ['PVZ',  'B'],
                 ['PVR',  'B'],
                 ['MBO',  'B'],                
                 # cortical subplate?
                 ['BMA',  'B'],
                 ['PA',  'B'],
                 # fiber tracts
                 ['cm', 'B'],  ### includes von, IIn, onl
                 ['pyd',  'B'],
                 ['cbp', 'B'],   ### cerebellar peduncles
                 


                 ]
            }, 
                       
        
        # ## CUPRIZONE + RECOVERY 3wks
        {'path':'/media/user/20TB_HDD_NAS_2/20240426_M313_REDO_REDO_MoE_CUP_6wks_RECOV_3wks_SHIELD_CUBIC_RIMS_RI_1493_after_PBwash_100p488_60p638_100msec/',
            'name': 'M313_REDO_REDO_CUP6wks_REC3wks_fused',     
            'exp':'Recovery',   'num':'M313',
            'sex':'F',           'age':120, 'thresh':0,  'weight':21, 'Exact age':120,     #500
            ### ANTS registered - but not brainreg --- yes brainreg
            
            'pkl_to_use': 'MYELIN',      ### for now, name not changed    
           # 'pkl_to_use': 'CUBIC',
             'exclude':[
                 ['VS',  'B'],
                 
                 # Hypothalamic midline regions
                 ['PVZ',  'B'],
                 ['PVR',  'B'],
                 ['MBO',  'B'],                
                 # cortical subplate?
                 ['BMA',  'B'],
                 ['PA',  'B'],
                 # fiber tracts
                 ['cm', 'B'],  ### includes von, IIn, onl
                 ['pyd',  'B'],
                 ['cbp', 'B'],   ### cerebellar peduncles
                 


                 ]
            },        
        
        
        ### 6wks cup + 3wks recovery
        ### A little bit wonky (lower) --- but RE-RAN with new MaskRCNN and looks good (more or less)
        
        ### HIGH IN VISUAL/AUD LAYER 1!!! --- REALLY MESSES WITH LAYER 1 --- likely due to bloody
        
        # {'path':'/media/user/20TB_HDD_NAS_2/20240416_M311_REDO_MoE_CUPRIZONE_6wks_RECOVERY_3wks_SHIELD_CUBIC_7d_RIMS_RI_1496_3d_100perc_488_80msec/',
        #     'name': 'M311_REDO_CUP6wks_REC3wks_fused',     
        #     'exp':'Recovery',   'num':'M311',
        #     'sex':'M',           'age':120, 'thresh':0,
            
        #     'pkl_to_use': 'MYELIN',      ### for now, name not changed    
        #    # 'pkl_to_use': 'CUBIC',
        #     },                          
      
    
        
       
        
        
        
        #%%% LncOL1
        # P60 - LncOL1
        #   {'path':'/media/user/20TB_HDD_NAS/20240209_M248_96rik_SHIELD_CUBIC_7d_RIMS_3d_RI_1493_sunflower/',
        #     'name': 'M248_96rik_P60_fused',     
        #     'exp':'LncOL1',     'num':'M248',
        #     'sex':'F',           'age':60, 'thresh':0,
        #     },   
         
        
        # P60 - LncOL1 with nanobodies
          {'path':'/media/user/8TB_HDD/20240216_M246_96rik_with_nanobodies_SHIELD_CUBIC_RIMS_RI_1493_sunflow/',
            'name': 'M246_96rik_nanobodies_P60_fused',     
            'exp':'LncOL1',   'num':'M246',
            
            'pkl_to_use': 'MYELIN',      ### for now, name not changed    
            
            'sex':'F',       'age':60, 'thresh':0,
            },   
         
        
        
        #   # ## 6 months control for 5xFAD - NO LOCATION
        #   #       {'path':'/media/user/Tx_LS_Data_1/20231218_M216_MoE_control_for_5xFAD_RIMS_RI_1489_sunflower_80perc/fused/',
        #   #        'name': 'M216_6months_fused',               
        #   #        'exp':'6mths',     'num':'M216',
        #   #        'sex':'M',         'age':180, 'thresh':0,
        #   #        },   
           
          
            
            #%%% P120
            
              # Cup control
            # {'path':'/media/user/8TB_HDD/20231116_M138_MoE_CasprtdT_Cup_CONTROL_6wk__SHIELD_RIMS_RI1487_5x_60perc_laser_SUNFLOWER/',
            #   'name': 'M138_cup_control_fused', 
            #   'exp':'P120_NEW_NEW',  'num':'M138',
            #   'sex':'M',     'age':120, 'thresh':500,
            #   },             
            
            
                
            # ### 3.5mos CONTROL for cup --- a little low... need to double check why... maybe CUPRIZONE CONTROL feed? - or FEMALE???
            # {'path':'/media/user/20TB_HDD_NAS/20240215_M264_Cup_CONTROL_3mos_SHIELD_DELIP_7d_RIMS_RI_1493_sunflow/',
            #   'name': 'M264_3mos_CUP_CONTROL_fused', 
            #   'exp':'P120_NEW',   'num':'M264',
            #   'sex':'F',      'age':62, 'thresh':400,
            #   'side':'R'
            #   },   
            
        
              
            # ### 3.5 months with Tie2Cre Ai9 --- ALSO P100
            # {'path':'/media/user/20TB_HDD_NAS/20240102_M226_MoE_Tie2Cre_Ai9_3mos_delip_RIMS_RI_14926_sunflow_80perc/',
            #   'name': 'M226_3mos_MoE_Tie2Cre_Ai9_fused',      
            #   'exp':'P120',  'num':'M226',
            #   'sex':'F',    'age':120, 'thresh':550,  ### is actual age of P100
            #   },   
            

            ## 4mos - tiny bit oily in minor areas --- was redone later again as well
            ### DECENT AMOUNT OF TISSUE TEARING AS WELL ### also a little bloody
            # {'path':'/media/user/20TB_HDD_NAS_2/20240416_M326_REDO_MoE_4mos_SHIELD_CUBIC_7d_RIMS_RI_1496_3d_sunflow_REDO_100perc_488laser_60perc_638laser_80msec/',
            #   'name': 'M326_REDO_4mos_fused',
            #   'exp':'P120_NEW',  'num':'M326',
            #   'sex':'F',    'age':120, 'thresh':500,  ### good
            # },
            
            

            # ## 4mos - seems okay, left hemisphere a bit smaller than right
            # {'path':'/media/user/20TB_HDD_NAS_2/20240423_M325_REDO_4mos_MoE_little_smaller_SHIELD_CUBIC_RIMS_RI_1496_4d_after_PB_wash_100perc_488_50perc_638_100msec_2/',
            #   'name': 'M325_REDO_4mos_fused',
            #   'exp':'P120_NEW',  'num':'M325',
            #   'sex':'F',    'age':120, 'thresh':500,  ### good
            # },
            

        #%%% P360
        # # # ## For 12 months - QUITE WARPED              
        # # # {'path':'/media/user/20TB_HDD_NAS_2/20240229_M305_MoE_12mos_SHIELD_CUBIC_6d_RIMS_2d_RI_new_14925_sunflow/',
        # # #   'name': 'M305_12mos_MoE_fused',            
        # # #   'exp':'P360',  'num':'M305',
        # # #   'sex':'M',     'age':360, 'thresh':600,
        # # #   },                            
                    
        # # #       ### For 12 months - with Tie2Cre;Ai9 - new stitching!!! --- LESS CELLS, why???
        # # # {'path':'/media/user/20TB_HDD_NAS/20240229_M274_MoE_Tie2Cre_Ai9_12mos_SHIELD_CUBIC_6d_RIMS_1d_RI_new_unsure_sunflow/',
        # # #   'name': 'M274_12mos_MoE_Tie2Cre_Ai9_fused',          
        # # #   'exp':'P360',  'num':'M274',
        # # #   'sex':'M',     'age':360, 'thresh':400,
        # # #   },      


        # # #       ## For 12 months --- CURRENTLY USING NORMALIZED SEG
        # # # {'path':'/media/user/20TB_HDD_NAS_2/20240420_M306_REDO_PB_washed_1_year_SHIELD_CUBIC_RI_RIMS_1496_1d_after_wash_100perc_488_50perc_638_100msec/',
        # # #   'name': 'M306_1yo_MoE_fused',          
        # # #   'exp':'P360',  'num':'M306',
        # # #   'sex':'M',     'age':360, 'thresh':500,
        # # #   },      


                 ]


    if mouse_num != 'all':
        
        
        specific_list = []
        for num in mouse_num:
            
            match_dict = dict(pd.DataFrame(metadata).iloc[np.where(pd.DataFrame(metadata)['num'] == num)[0][0]])
            specific_list.append(match_dict)
        
        metadata=specific_list

    return metadata



