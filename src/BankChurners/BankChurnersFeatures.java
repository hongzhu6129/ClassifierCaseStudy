package BankChurners;

public final class BankChurnersFeatures {
//	public final static int Attrition_Flag = 0; /* {"Existing Customer":0,"Attrited Customer":1} */
	
	//The indexes of finite discrete features:
	public final static int Gender =0; /*{'F':0, 'M':1} */
	public final static int Education_Level = 1; /* {'Graduate':0, 'High School':1, 'Uneducated':2, 'College':3, 'Unknown':4, 'Post-Graduate':5, 'Doctorate':6} */
	public final static int Marital_Status =2; /* {'Married':0, 'Single':1, 'Divorced':2, 'Unknown':3} */
	public final static int Income_Category =3; /* {'Less than $40K':0, '$40K - $60K':1, '$60K - $80K':2, '$80K - $120K':3, '$120K +':4, 'Unknown':5} */
	public final static int Card_Category = 4; /* {'Blue':0, 'Silver':1, 'Gold':2, 'Platinum':3} */

	//The domain sizes of finite discrete features
	public final static int[] size = {2, 7, 4, 6, 4}; 
	
	//The indexes of infinite discrete features (integers): 
	public final static int Customer_Age =0; /* min: 26, max:73, range:[20,80], step: 3 */
	public final static int Dependent_count =1;  /* min:0, max: 5, range:[0,10], step: 1 */
	public final static int Months_on_book =2; /*min:13, max:56, range:[0,100], step:5 */
	public final static int Total_Relationship_Count =3; /* min:1, max:6, range:[0,10], step: 1 */
	public final static int Months_Inactive_12_mon =4; /* min:0, max 6, range:[0, 10], step: 1 */
	public final static int Contacts_Count_12_mon =5; /* min:0, max: 6, range: [0, 10], step: 1 */
	public final static int Credit_Limit =6; /* min: 1440, max: 34500, range: [1000, 40000], step: 2000 */
	public final static int Total_Revolving_Bal =7; /* min: 0, max: 2517, range: [0, 3000], step: 150 */
	public final static int Avg_Open_To_Buy =8; /* min: 3, max: 34500, range: [0, 40000], step: 2000 */
	public final static int Total_Trans_Amt =9; /* min 510, max: 18500, range: [0, 20000], step: 1000 */
	public final static int Total_Trans_Ct =10; /* min: 10, max: 139, range: [0, 200], step: 10 */
	
	public final static int[] intLower = {20, 0, 0, 0, 0, 0, 1000, 0, 0, 0, 0};
	public final static int[] intUpper = {80, 10, 100, 10, 10, 10, 40000, 3000, 40000, 20000, 200};
	public final static int[] intStep = {3, 1, 5, 1, 1, 1, 2000, 150, 2000, 1000, 10};
	
	//The indexes of real number features:
	public final static int Total_Amt_Chng_Q4_Q1 =0; /* min: 0, max: 3.4, range: [0, 10], step: 0.5 */
	public final static int Total_Ct_Chng_Q4_Q1 =1; /* min: 0, max: 3.71, range: [0, 10], step: 0.5 */
	public final static int Avg_Utilization_Ratio =2; /* min: 0, max: 1, range: [0, 1], step: 0.05 */
	
	public final static double[] realLower = {0, 0, 0};
	public final static double[] realUpper = {10, 10, 1};
	public final static double[] realStep = {0.5, 0.5, 0.05};
}
