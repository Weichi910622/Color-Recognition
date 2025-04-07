#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "NUC100Series.h"

#define PLL_CLOCK       50000000

/******************************************************************
 * dataset format setting
 ******************************************************************/

#define train_data_num 168  				//Total number of training data
#define test_data_num 42					//Total number of testing data

/******************************************************************
 * Network Configuration - customized per network 
 ******************************************************************/
#define input_length 3						//The number of input
#define HiddenNodes 20 						//The number of neurons in hidden layer
#define target_num 7 						//The number of output

const float LearningRate = 	0.1;					//Learning Rate
const float Momentum = 0.9;
const float InitialWeightMax = 0.5;
const float goal_acc = 0.95;						//Target accuracy


// Create training dataset/output
float train_data_input[train_data_num][input_length] = {{891, 865, 1286}, {937, 928, 1370}, {856, 832, 1262}, {922, 913, 1357}, {870, 836, 1258}, {947, 939, 1374}, {867, 833, 1257}, {934, 924, 1367}, {918, 898, 1325}, {873, 843, 1264}, {894, 883, 1318}, {929, 910, 1334}, {907, 875, 1302}, {945, 938, 1373}, {861, 839, 1269}, {871, 849, 1285}, {908, 896, 1341}, {931, 925, 1364}, {922, 900, 1326}, {947, 932, 1369}, {940, 935, 1372}, {860, 837, 1263}, {921, 910, 1352}, {901, 884, 1330},
{346, 347, 798}, {329, 329, 769}, {320, 325, 761}, {408, 416, 882}, {416, 429, 888}, {442, 440, 891}, {373, 394, 867}, {476, 434, 807}, {377, 357, 788}, {400, 378, 807}, {335, 349, 797}, {300, 317, 781}, {305, 345, 850}, {312, 362, 885}, {315, 374, 900}, {321, 370, 891}, {316, 355, 867}, {325, 354, 854}, {305, 326, 795}, {298, 321, 767}, {307, 325, 778}, {300, 315, 748}, {299, 313, 737}, {306, 318, 740},
{689, 553, 928}, {601, 487, 870}, {607, 477, 850}, {597, 478, 860}, {571, 450, 818}, {593, 471, 827}, {591, 470, 820}, {582, 461, 811}, {613, 490, 845}, {600, 484, 854}, {596, 477, 850}, {619, 497, 886}, {605, 485, 873}, {601, 480, 865}, {616, 502, 903}, {609, 489, 885}, {594, 480, 868}, {596, 478, 867}, {620, 504, 902}, {603, 481, 869}, {609, 496, 894}, {615, 494, 872}, {650, 516, 868}, {676, 536, 891},
{605, 485, 821}, {615, 494, 843}, {605, 492, 844}, {602, 476, 820}, {636, 508, 863}, {636, 512, 864}, {606, 487, 839}, {612, 495, 838}, {617, 498, 849}, {627, 502, 844}, {628, 499, 851}, {624, 503, 891}, {637, 533, 976}, {643, 515, 892}, {626, 504, 854}, {604, 470, 786}, {638, 493, 806}, {633, 493, 800}, {656, 509, 817}, {646, 500, 816}, {633, 491, 797}, {638, 497, 808}, {654, 507, 821}, {651, 500, 815},
{763, 699, 1094}, {727, 642, 1007}, {735, 644, 992}, {717, 638, 998}, {721, 621, 932}, {666, 562, 870}, {727, 621, 923}, {718, 604, 889}, {689, 585, 861}, {717, 608, 890}, {734, 619, 908}, {734, 624, 899}, {734, 618, 893}, {768, 649, 924}, {749, 638, 915}, {740, 627, 903}, {733, 622, 898}, {736, 622, 905}, {701, 593, 873}, {708, 599, 894}, {749, 643, 955}, {758, 647, 952}, {735, 621, 898}, {772, 658, 930},
{770, 763, 1104}, {725, 722, 1073}, {758, 755, 1104}, {728, 724, 1071}, {736, 735, 1078}, {767, 766, 1097}, {733, 729, 1075}, {779, 776, 1124}, {784, 795, 1142}, {771, 776, 1126}, {754, 758, 1095}, {704, 698, 1009}, {766, 768, 1078}, {768, 766, 1075}, {802, 809, 1119}, {804, 803, 1126}, {776, 777, 1101}, {796, 802, 1145}, {845, 855, 1189}, {775, 776, 1121}, {759, 759, 1101}, {753, 750, 1069}, {763, 768, 1078}, {784, 786, 1097},
{508, 525, 951}, {401, 410, 808}, {391, 416, 821}, {343, 359, 742}, {337, 348, 711}, {349, 373, 720}, {344, 374, 729}, {334, 380, 742}, {335, 377, 766}, {325, 362, 754}, {353, 392, 812}, {374, 429, 866}, {339, 386, 790}, {340, 399, 770}, {340, 390, 757}, {333, 388, 745}, {320, 363, 738}, {338, 391, 818}, {351, 410, 861}, {330, 386, 779}, {329, 377, 771}, {332, 379, 772}, {333, 384, 763}, {335, 384, 756}
};	// You can put your training dataset here
int train_data_output[train_data_num][target_num] = {
{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},
{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},
{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},
{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},
{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},
{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},
{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1}}; 		// Label of the training data

// Create testing dataset/output
float test_data_input[test_data_num][input_length] = {{943, 936, 1372}, {942, 925, 1356}, {860, 824, 1255}, {863, 827, 1253}, {911, 900, 1342}, {899, 878, 1300},
{347, 381, 886}, {440, 408, 796}, {451, 466, 949}, {358, 371, 825}, {320, 337, 791}, {311, 336, 785},
{643, 510, 857}, {636, 514, 856}, {638, 509, 855}, {673, 546, 891}, {686, 548, 900}, {692, 553, 916},
{656, 514, 828}, {660, 517, 829}, {658, 508, 826}, {652, 505, 812}, {668, 522, 828}, {687, 536, 851},
{807, 686, 957}, {814, 697, 966}, {813, 686, 958}, {800, 679, 946}, {774, 654, 929}, {758, 643, 915},
{771, 776, 1087}, {801, 808, 1125}, {767, 774, 1087}, {788, 789, 1105}, {796, 797, 1112}, {820, 821, 1135},
{371, 417, 833}, {349, 396, 839}, {324, 372, 762}, {331, 383, 754}, {331, 378, 745}, {331, 382, 749}};		// You can put your testing dataset here
int test_data_output[test_data_num][target_num] = {
{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},{1,0,0,0,0,0,0},
{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},{0,1,0,0,0,0,0},
{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},{0,0,1,0,0,0,0},
{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},{0,0,0,1,0,0,0},
{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},{0,0,0,0,1,0,0},
{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},{0,0,0,0,0,1,0},
{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1},{0,0,0,0,0,0,1}}; 			// Label of the testing data

/******************************************************************
 * End Network Configuration
 ******************************************************************/


int ReportEvery10;
int RandomizedIndex[train_data_num];
long  TrainingCycle;
float Rando;
float Error;
float Accum;

float data_mean[3] ={0};
float data_std[3] ={0};

float Hidden[HiddenNodes];
float Output[target_num];
float HiddenWeights[input_length+1][HiddenNodes];
float OutputWeights[HiddenNodes+1][target_num];
float HiddenDelta[HiddenNodes];
float OutputDelta[target_num];
float ChangeHiddenWeights[input_length+1][HiddenNodes];
float ChangeOutputWeights[HiddenNodes+1][target_num];

int target_value;
int out_value;
int max;


/*---------------------------------------------------------------------------------------------------------*/
/* Define Function Prototypes                                                                              */
/*---------------------------------------------------------------------------------------------------------*/
void SYS_Init(void);
void UART0_Init(void);
void AdcSingleCycleScanModeTest(void);


void SYS_Init(void)
{
    /*---------------------------------------------------------------------------------------------------------*/
    /* Init System Clock                                                                                       */
    /*---------------------------------------------------------------------------------------------------------*/

    /* Enable Internal RC 22.1184MHz clock */
    CLK_EnableXtalRC(CLK_PWRCON_OSC22M_EN_Msk);

    /* Waiting for Internal RC clock ready */
    CLK_WaitClockReady(CLK_CLKSTATUS_OSC22M_STB_Msk);

    /* Switch HCLK clock source to Internal RC and HCLK source divide 1 */
    CLK_SetHCLK(CLK_CLKSEL0_HCLK_S_HIRC, CLK_CLKDIV_HCLK(1));

    /* Enable external XTAL 12MHz clock */
    CLK_EnableXtalRC(CLK_PWRCON_XTL12M_EN_Msk);

    /* Waiting for external XTAL clock ready */
    CLK_WaitClockReady(CLK_CLKSTATUS_XTL12M_STB_Msk);

    /* Set core clock as PLL_CLOCK from PLL */
    CLK_SetCoreClock(PLL_CLOCK);

    /* Enable UART module clock */
    CLK_EnableModuleClock(UART0_MODULE);

    /* Enable ADC module clock */
    CLK_EnableModuleClock(ADC_MODULE);

    /* Select UART module clock source */
    CLK_SetModuleClock(UART0_MODULE, CLK_CLKSEL1_UART_S_PLL, CLK_CLKDIV_UART(1));

    /* ADC clock source is 22.1184MHz, set divider to 7, ADC clock is 22.1184/7 MHz */
    CLK_SetModuleClock(ADC_MODULE, CLK_CLKSEL1_ADC_S_HIRC, CLK_CLKDIV_ADC(7));

    /*---------------------------------------------------------------------------------------------------------*/
    /* Init I/O Multi-function                                                                                 */
    /*---------------------------------------------------------------------------------------------------------*/

    /* Set GPB multi-function pins for UART0 RXD and TXD */
    SYS->GPB_MFP &= ~(SYS_GPB_MFP_PB0_Msk | SYS_GPB_MFP_PB1_Msk);
    SYS->GPB_MFP |= SYS_GPB_MFP_PB0_UART0_RXD | SYS_GPB_MFP_PB1_UART0_TXD;

    /* Disable the GPA0 - GPA3 digital input path to avoid the leakage current. */
    GPIO_DISABLE_DIGITAL_PATH(PA, 0xF);

    /* Configure the GPA0 - GPA3 ADC analog input pins */
    SYS->GPA_MFP &= ~(SYS_GPA_MFP_PA0_Msk | SYS_GPA_MFP_PA1_Msk | SYS_GPA_MFP_PA2_Msk | SYS_GPA_MFP_PA3_Msk) ;
    SYS->GPA_MFP |= SYS_GPA_MFP_PA0_ADC0 | SYS_GPA_MFP_PA1_ADC1 | SYS_GPA_MFP_PA2_ADC2 | SYS_GPA_MFP_PA3_ADC3 ;
    SYS->ALT_MFP1 = 0;

}

/*---------------------------------------------------------------------------------------------------------*/
/* Init UART                                                                                               */
/*---------------------------------------------------------------------------------------------------------*/
void UART0_Init()
{
    /* Reset IP */
    SYS_ResetModule(UART0_RST);

    /* Configure UART0 and set UART0 Baudrate */
    UART_Open(UART0, 115200);
}

void scale_data()
{
		float sum[3] = {0};
		int i, j;
		
		// Compute Data Mean
		for(i = 0; i < train_data_num; i++){
			for(j = 0; j < input_length; j++){
				sum[j] += train_data_input[i][j];
			}
		}
		for(j = 0; j < input_length ; j++){
			data_mean[j] = sum[j] / train_data_num;
			printf("MEAN: %.2f\n", data_mean[j]);
			sum[j] = 0.0;
		}
		
		// Compute Data STD
		for(i = 0; i < train_data_num; i++){
			for(j = 0; j < input_length ; j++){
				sum[j] += pow(train_data_input[i][j] - data_mean[j], 2);
			}
		}
		for(j = 0; j < input_length; j++){
			data_std[j] = sqrt(sum[j]/train_data_num);
			printf("STD: %.2f\n", data_std[j]);
			sum[j] = 0.0;
		}
}

void normalize(float *data)
{
		int i;
	
		for(i = 0; i < input_length; i++){
			data[i] = (data[i] - data_mean[i]) / data_std[i];
		}
}

int train_preprocess()
{
    int i;
    
    for(i = 0 ; i < train_data_num ; i++)
    {
        normalize(train_data_input[i]);
    }
		
    return 0;
}

int test_preprocess()
{
    int i;

    for(i = 0 ; i < test_data_num ; i++)
    {
        normalize(test_data_input[i]);
    }
		
    return 0;
}

int data_setup()
{
    int i;
		//int j;
		int p, ret;
		unsigned int seed = 1;
	
		/* Set the ADC operation mode as single-cycle, input mode as single-end and
                 enable the analog input channel 0, 1 and 2 */
    ADC_Open(ADC, ADC_ADCR_DIFFEN_SINGLE_END, ADC_ADCR_ADMD_SINGLE_CYCLE, 0x7);

    /* Power on ADC module */
    ADC_POWER_ON(ADC);

    /* Clear the A/D interrupt flag for safe */
    ADC_CLR_INT_FLAG(ADC, ADC_ADF_INT);

    /* Start A/D conversion */
    ADC_START_CONV(ADC);

    /* Wait conversion done */
    while(!ADC_GET_INT_FLAG(ADC, ADC_ADF_INT));
		
		for(i = 0; i < 3; i++)
    {
				seed *= ADC_GET_CONVERSION_DATA(ADC, i);
    }
		seed *= 1000;
		printf("\nRandom seed: %d\n", seed);
    srand(seed);

    ReportEvery10 = 1;
    for( p = 0 ; p < train_data_num ; p++ ) 
    {    
        RandomizedIndex[p] = p ;
    }
		
		scale_data();
    ret = train_preprocess();
    ret |= test_preprocess();
    if(ret) //Error Check
        return 1;

    return 0;
}

void run_train_data()
{
    int i, j, p;
    int correct=0;
    float accuracy = 0;
    printf("Train result:\n");
    for( p = 0 ; p < train_data_num ; p++ )
    { 
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (train_data_output[p][i] > train_data_output[p][max]) {
                max = i;
            }
        }
        target_value = max;
        
    /******************************************************************
    * Compute hidden layer activations
    ******************************************************************/

        for( i = 0 ; i < HiddenNodes ; i++ ) {    
            Accum = HiddenWeights[input_length][i] ;
            for( j = 0 ; j < input_length ; j++ ) {
                Accum += train_data_input[p][j] * HiddenWeights[j][i] ;
            }
            Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

    /******************************************************************
    * Compute output layer activations and calculate errors
    ******************************************************************/

        for( i = 0 ; i < target_num ; i++ ) {    
            Accum = OutputWeights[HiddenNodes][i] ;
            for( j = 0 ; j < HiddenNodes ; j++ ) {
                Accum += Hidden[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
        }
        
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (Output[i] > Output[max]) {
                max = i;
            }
        }
        out_value = max;

        if(out_value!=target_value)
            printf("Error --> Training Pattern: %d,Target : %d, Output : %d\n", p, target_value, out_value);
        else
            correct++;
        }
        // Calculate accuracy
        accuracy = (float)correct / train_data_num;
        printf ("Accuracy = %.2f /100 \n",accuracy*100);

}

void run_test_data()
{
    int i, j, p;
    int correct=0;
    float accuracy = 0;
    printf("Test result:\n");
    for( p = 0 ; p < test_data_num ; p++ )
    { 
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (test_data_output[p][i] > test_data_output[p][max]) {
                max = i;
            }
        }
        target_value = max;
        
    /******************************************************************
    * Compute hidden layer activations
    ******************************************************************/

        for( i = 0 ; i < HiddenNodes ; i++ ) {    
            Accum = HiddenWeights[input_length][i] ;
            for( j = 0 ; j < input_length ; j++ ) {
                Accum += test_data_input[p][j] * HiddenWeights[j][i] ;
            }
            Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

    /******************************************************************
    * Compute output layer activations and calculate errors
    ******************************************************************/

        for( i = 0 ; i < target_num ; i++ ) {    
            Accum = OutputWeights[HiddenNodes][i] ;
            for( j = 0 ; j < HiddenNodes ; j++ ) {
                Accum += Hidden[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
        }
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (Output[i] > Output[max]) {
                max = i;
            }
        }
        out_value = max;

        if(out_value!=target_value)
            printf("Error --> Training Pattern: %d,Target : %d, Output : %d\n", p, target_value, out_value);
        else
            correct++;
        }
        // Calculate accuracy
        accuracy = (float)correct / test_data_num;
        printf ("Accuracy = %.2f /100 \n",accuracy*100);
}

float Get_Train_Accuracy()
{
    int i, j, p;
    int correct = 0;
		float accuracy = 0;
    for (p = 0; p < train_data_num; p++)
    {
/******************************************************************
* Compute hidden layer activations
******************************************************************/

        for( i = 0 ; i < HiddenNodes ; i++ ) {    
            Accum = HiddenWeights[input_length][i] ;
            for( j = 0 ; j < input_length ; j++ ) {
                Accum += train_data_input[p][j] * HiddenWeights[j][i] ;
            }
            Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
        }

/******************************************************************
* Compute output layer activations
******************************************************************/

        for( i = 0 ; i < target_num ; i++ ) {    
            Accum = OutputWeights[HiddenNodes][i] ;
            for( j = 0 ; j < HiddenNodes ; j++ ) {
                Accum += Hidden[j] * OutputWeights[j][i] ;
            }
            Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
        }
        //get target value
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (train_data_output[p][i] > train_data_output[p][max]) {
                max = i;
            }
        }
        target_value = max;
        //get output value
        max = 0;
        for (i = 1; i < target_num; i++) 
        {
            if (Output[i] > Output[max]) {
                max = i;
            }
        }
        out_value = max;
        //compare output and target
        if (out_value==target_value)
        {
            correct++;
        }
    }

    // Calculate accuracy
    accuracy = (float)correct / train_data_num;
    return accuracy;
}

void load_weight()
{
    int i,j;
    printf("\n=======Hidden Weight=======\n");
    printf("{");
    for(i = 0; i <= input_length ; i++)
    {
        printf("{");
        for (j = 0; j < HiddenNodes; j++)
        {
            if(j!=HiddenNodes-1){
                printf("%f,", HiddenWeights[i][j]);
            }else{
                printf("%f", HiddenWeights[i][j]);
            }
        }
        if(i!=input_length){
            printf("},\n");
        }else {
            printf("}");
        }
    }
    printf("}\n");

    printf("\n=======Output Weight=======\n");

    for(i = 0; i <= HiddenNodes ; i++)
    {
        printf("{");
        for (j = 0; j < target_num; j++)
        {
            if(j!=target_num-1){
                printf("%f,", OutputWeights[i][j]);
            }else{
                printf("%f", OutputWeights[i][j]);
            }
        }
        if(i!=HiddenNodes){
            printf("},\n");
        }else {
            printf("}");
        }
    }
    printf("}\n");
}

void AdcSingleCycleScanModeTest()
{
		int i, j;
    uint32_t u32ChannelCount;
    float single_data_input[3];
		char output_string[10] = {NULL};

    printf("\n");	
		printf("[Phase 3] Start Prediction ...\n\n");
		PB3=1;
    while(1)
    {
			
				/* Set the ADC operation mode as single-cycle, input mode as single-end and
                 enable the analog input channel 0, 1, 2 and 3 */
        ADC_Open(ADC, ADC_ADCR_DIFFEN_SINGLE_END, ADC_ADCR_ADMD_SINGLE_CYCLE, 0x7);

        /* Power on ADC module */
        ADC_POWER_ON(ADC);

        /* Clear the A/D interrupt flag for safe */
        ADC_CLR_INT_FLAG(ADC, ADC_ADF_INT);

        /* Start A/D conversion */
        ADC_START_CONV(ADC);

        /* Wait conversion done */
        while(!ADC_GET_INT_FLAG(ADC, ADC_ADF_INT));

        for(u32ChannelCount = 0; u32ChannelCount < 3; u32ChannelCount++)
        {
            single_data_input[u32ChannelCount] = ADC_GET_CONVERSION_DATA(ADC, u32ChannelCount);
        }
				normalize(single_data_input);
						

				// Compute hidden layer activations
				for( i = 0 ; i < HiddenNodes ; i++ ) {    
						Accum = HiddenWeights[input_length][i] ;
						for( j = 0 ; j < input_length ; j++ ) {
								Accum += single_data_input[j] * HiddenWeights[j][i] ;
						}
						Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
				}

				// Compute output layer activations
				for( i = 0 ; i < target_num ; i++ ) {    
						Accum = OutputWeights[HiddenNodes][i] ;
						for( j = 0 ; j < HiddenNodes ; j++ ) {
								Accum += Hidden[j] * OutputWeights[j][i] ;
						}
						Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
				}
						
				max = 0;
				for (i = 1; i < target_num; i++) 
				{
						if (Output[i] > Output[max]) {
								max = i;
						}
				}
				out_value = max;
				
				switch(out_value){
						case 0:
								strcpy(output_string, "Ambient");
								break;
						case 1:
								strcpy(output_string, "Blue");	
								break;
						case 2:
								strcpy(output_string, "Magenta");	
								break;
						case 3:
								strcpy(output_string, "Red");
								break;
						case 4:
								strcpy(output_string, "Orange");	
								break;
						case 5:
								strcpy(output_string, "Yellow");	
								break;
						case 6:
								strcpy(output_string, "Green");
								break;
				}
				
				printf("\rPrediction output: %-8s", output_string);
				CLK_SysTickDelay(500000);


    }
}

/*---------------------------------------------------------------------------------------------------------*/
/* MAIN function                                                                                           */
/*---------------------------------------------------------------------------------------------------------*/

int main(void)
{
		int i, j, p, q, r;
    float accuracy=0;

    /* Unlock protected registers */
    SYS_UnlockReg();

    /* Init System, IP clock and multi-function I/O */
    SYS_Init();

    /* Lock protected registers */
    SYS_LockReg();

    /* Init UART0 for printf */
    UART0_Init();
	
	  PB->PMD = (PB->PMD & (~GPIO_PMD_PMD3_Msk))|(GPIO_PMD_OUTPUT << GPIO_PMD_PMD3_Pos);
	  PB3=0;
	
		printf("\n+-----------------------------------------------------------------------+\n");
    printf("|                        LAB8 - Machine Learning                        |\n");
    printf("+-----------------------------------------------------------------------+\n");
		printf("System clock rate: %d Hz\n", SystemCoreClock);

    printf("\n[Phase 1] Initialize DataSet ...");
	  /* Data Init (Input / Output Preprocess) */
		if(data_setup()){
        printf("[Error] Datasets Setup Error\n");
        return 0;
    }else
				printf("Done!\n\n");
		
		printf("[Phase 2] Start Model Training ...\n");
		// Initialize HiddenWeights and ChangeHiddenWeights 
    for( i = 0 ; i < HiddenNodes ; i++ ) {    
        for( j = 0 ; j <= input_length ; j++ ) { 
            ChangeHiddenWeights[j][i] = 0.0 ;
            Rando = (float)((rand() % 100))/100;
            HiddenWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }

    // Initialize OutputWeights and ChangeOutputWeights
    for( i = 0 ; i < target_num ; i ++ ) {    
        for( j = 0 ; j <= HiddenNodes ; j++ ) {
            ChangeOutputWeights[j][i] = 0.0 ;  
            Rando = (float)((rand() % 100))/100;        
            OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
        }
    }

    // Begin training 
    for(TrainingCycle = 1 ; TrainingCycle < 2147483647 ; TrainingCycle++)
    {
        Error = 0.0 ;

        // Randomize order of training patterns
        for( p = 0 ; p < train_data_num ; p++) {
            q = rand()%train_data_num;
            r = RandomizedIndex[p] ; 
            RandomizedIndex[p] = RandomizedIndex[q] ; 
            RandomizedIndex[q] = r ;
        }

        // Cycle through each training pattern in the randomized order
        for( q = 0 ; q < train_data_num ; q++ ) 
        {    
            p = RandomizedIndex[q];

            // Compute hidden layer activations
            for( i = 0 ; i < HiddenNodes ; i++ ) {    
                Accum = HiddenWeights[input_length][i] ;
                for( j = 0 ; j < input_length ; j++ ) {
                    Accum += train_data_input[p][j] * HiddenWeights[j][i] ;
                }
                Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
            }

            // Compute output layer activations and calculate errors
            for( i = 0 ; i < target_num ; i++ ) {    
                Accum = OutputWeights[HiddenNodes][i] ;
                for( j = 0 ; j < HiddenNodes ; j++ ) {
                    Accum += Hidden[j] * OutputWeights[j][i] ;
                }
                Output[i] = 1.0/(1.0 + exp(-Accum)) ;   
                OutputDelta[i] = (train_data_output[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]) ;   
                Error += 0.5 * (train_data_output[p][i] - Output[i]) * (train_data_output[p][i] - Output[i]) ;
            }

            // Backpropagate errors to hidden layer
            for( i = 0 ; i < HiddenNodes ; i++ ) {    
                Accum = 0.0 ;
                for( j = 0 ; j < target_num ; j++ ) {
                    Accum += OutputWeights[i][j] * OutputDelta[j] ;
                }
                HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
            }

            // Update Input-->Hidden Weights
            for( i = 0 ; i < HiddenNodes ; i++ ) {     
                ChangeHiddenWeights[input_length][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[input_length][i] ;
                HiddenWeights[input_length][i] += ChangeHiddenWeights[input_length][i] ;
                for( j = 0 ; j < input_length ; j++ ) { 
                    ChangeHiddenWeights[j][i] = LearningRate * train_data_input[p][j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
                    HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;
                }
            }

            // Update Hidden-->Output Weights
            for( i = 0 ; i < target_num ; i ++ ) {    
                ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i] ;
                OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i] ;
                for( j = 0 ; j < HiddenNodes ; j++ ) {
                    ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
                    OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
                }
            }
        }
        accuracy = Get_Train_Accuracy();

        // Every 10 cycles send data to terminal for display
        ReportEvery10 = ReportEvery10 - 1;
        if (ReportEvery10 == 0)
        {
            
            printf ("\nTrainingCycle: %ld\n",TrainingCycle);
            printf ("Error = %.5f\n",Error);
            printf ("Accuracy = %.2f /100 \n",accuracy*100);
            //run_train_data();

            if (TrainingCycle==1)
            {
                ReportEvery10 = 9;
            }
            else
            {
                ReportEvery10 = 10;
            }
        }

        // If error rate is less than pre-determined threshold then end
        if( accuracy >= goal_acc ) break ;
    }

    printf ("\nTrainingCycle: %ld\n",TrainingCycle);
    printf ("Error = %.5f\n",Error);
    run_train_data();
    printf ("Training Set Solved!\n");
    printf ("--------\n"); 
    printf ("Testing Start!\n ");
    run_test_data();
    printf ("--------\n"); 
    ReportEvery10 = 1;
    load_weight();
		
		printf("\nModel Training Phase has ended.\n");

    /* Start prediction */
    AdcSingleCycleScanModeTest();

    while(1);
}
