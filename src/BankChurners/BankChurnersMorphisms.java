package BankChurners;

import MLTestFramework.*;

//Datamorphic Test Specification 
import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Random;
import javax.swing.JFileChooser;

import morphy.annotations.*;  
public class BankChurnersMorphisms extends MLTest<BankChurnersValue, Integer>{
	
	@TestSetContainer(
		inputTypeName = "BankChurnersValue",
		outputTypeName = "Integer")
	public TestPool<BankChurnersValue, Integer> tSuite = testSuite;
 
	@MakeSeed
	public void GenerateRandomTestCases(){	
		Random randomGenerator = new Random();
		TestCase<BankChurnersValue, Integer> tc;
		for (int i=0; i<100;i++){
			tc = new TestCase<BankChurnersValue, Integer>();;
			BankChurnersValue bc = new BankChurnersValue();
			for (int j=0; j<=4; j++) {
				bc.discValue[j]=randomGenerator.nextInt(BankChurnersFeatures.size[j]);
			}
			for (int j=0; j<=10; j++) {
				bc.intValue[j]=randomGenerator.nextInt();
			}
			for (int j=0; j<=2; j++) {
				bc.realValue[j] = randomGenerator.nextDouble();
			}
			tc.input = bc;
			tc.setFeature(TestDataFeature.original);
			tc.setType("Random");
			testSuite.addTestCase(tc);
		}
	}

	@MakeSeed
	public void readTestCasesFromCSVFile(){
		BankChurnersValue bc;
		TestCase<BankChurnersValue, Integer> tc;
		String directory = System.getProperty("user.dir");
		JFileChooser fileChooser = new JFileChooser(directory);
		String fileName;
		File file; 
		if (fileChooser.showOpenDialog(null)==JFileChooser.APPROVE_OPTION) {
			file = fileChooser.getSelectedFile();
			fileName = file.getName();
		}else {
			return; 
		};
		FileReader fileReader;
		try {
			fileReader = new FileReader(file);
		} catch (FileNotFoundException e1) {
			return;
		};
		BufferedReader reader = new BufferedReader(fileReader);
		String line;
		try {
			while ((line = reader.readLine()) != null) {
				tc = new TestCase<BankChurnersValue, Integer>();;
				bc = new BankChurnersValue();
				String[] numStrs = line.split(",");
				int numValue = 0; 
				switch (numStrs[BankChurnersFeatures.Gender]) {
					case "F": numValue = 0; break; 
					case "M": numValue = 1; break;
				}		
				bc.discValue[BankChurnersFeatures.Gender] = numValue; 
				
				numValue = 0; 
				switch (numStrs[BankChurnersFeatures.Education_Level]) {
					case "Graduate": numValue = 0; break; 
					case "High School": numValue = 1; break;
					case "Uneducated": numValue = 2; break;
					case "College": numValue = 3; break; 
					case "Unknown": numValue = 4; break; 
					case "Post-Graduate": numValue = 5; break; 
					case "Doctorate": numValue = 6; break; 
				}		
				bc.discValue[BankChurnersFeatures.Education_Level] = numValue; 
		
				numValue = 0; 
				switch (numStrs[BankChurnersFeatures.Marital_Status]) {
					case "Married": numValue = 0; break; 
					case "Single": numValue = 1; break;
					case "Divorced": numValue = 2; break;
					case "Unknown": numValue = 3; break; 
				}		
				bc.discValue[BankChurnersFeatures.Marital_Status] = numValue;
				
				numValue = 0; 
				switch (numStrs[BankChurnersFeatures.Income_Category]) {
					case "Less than $40K": numValue = 0; break; 
					case "$40K - $60K": numValue = 1; break;
					case "$60K - $80K": numValue = 2; break;
					case "$80K - $120K": numValue = 3; break;
					case "$120K +": numValue = 4; break;
					case "Unknown": numValue = 5; break;
				}		
				bc.discValue[BankChurnersFeatures.Income_Category] = numValue;
				
				for(int i=0; i<=10; i++) {
					bc.intValue[i] = Integer.valueOf(numStrs[i+5]);
				}
				
				for(int i=0; i<=2; i++) {
					bc.realValue[i] = Double.valueOf(numStrs[i+16]);
				}

				tc.input = bc;
				switch (numStrs[19]) {
					case "Existing Customer": tc.output = 0; break;
					case "Attrited Customer": tc.output = 1; break;
				}
				
				tc.setFeature(TestDataFeature.original);
				tc.setType("fromFile");
				testSuite.addTestCase(tc);
			}
		} catch (IOException e2) {
			e2.printStackTrace();
			System.out.println("Error in reading file.");
			return;
		}
		try {
			reader.close();
		} catch (IOException e) {
			e.printStackTrace();
			return;
		}
	}
	
	public TestCase<BankChurnersValue, Integer> discFeatureUp (int feature, TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = new TestCase<BankChurnersValue, Integer>();
		BankChurnersValue bc = new BankChurnersValue();
		for (int i=0; i<=4; i++) {
			bc.discValue[i] = seed.input.discValue[i];
		};
		if (seed.input.discValue[feature] < BankChurnersFeatures.size[feature]-2) {
			bc.discValue[feature] ++;
		}
		mutant.input = bc;
		return mutant; 
	}
	
	public TestCase<BankChurnersValue, Integer> intFeatureUp (int feature, TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = new TestCase<BankChurnersValue, Integer>();
		BankChurnersValue bc = new BankChurnersValue();
		for (int i=0; i<=10; i++) {
			bc.intValue[i] = seed.input.intValue[i];
		};
		bc.intValue[feature] += BankChurnersFeatures.intStep[feature];
		if (bc.intValue[feature] > BankChurnersFeatures.intUpper[feature]){
			bc.intValue[feature] = BankChurnersFeatures.intUpper[feature];
		}
		mutant.input = bc;
		return mutant; 
	}
	
	public TestCase<BankChurnersValue, Integer> realFeatureUp (int feature, TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = new TestCase<BankChurnersValue, Integer>();
		BankChurnersValue bc = new BankChurnersValue();
		for (int i=0; i<=2; i++) {
			bc.realValue[i] = seed.input.realValue[i];
		};
		bc.realValue[feature] += BankChurnersFeatures.realStep[feature];
		if (bc.realValue[feature] > BankChurnersFeatures.realUpper[feature]) {
			bc.realValue[feature] = BankChurnersFeatures.realUpper[feature];
		}
		mutant.input = bc;
		return mutant; 
	}
	
	public TestCase<BankChurnersValue, Integer> discFeatureDown (int feature, TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = new TestCase<BankChurnersValue, Integer>();
		BankChurnersValue bc = new BankChurnersValue();
		for (int i=0; i<=4; i++) {
			bc.discValue[i] = seed.input.discValue[i];
		};
		if (seed.input.discValue[feature] > 0) {
			bc.discValue[feature]--;
		}
		mutant.input = bc;
		return mutant; 
	}
	
	public TestCase<BankChurnersValue, Integer> intFeatureDown (int feature, TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = new TestCase<BankChurnersValue, Integer>();
		BankChurnersValue bc = new BankChurnersValue();
		for (int i=0; i<=10; i++) {
			bc.intValue[i] = seed.input.intValue[i];
		};
		bc.intValue[feature] -= BankChurnersFeatures.intStep[feature];
		if (bc.intValue[feature] < BankChurnersFeatures.intLower[feature]) {
			bc.intValue[feature] = BankChurnersFeatures.intLower[feature];
		}
		mutant.input = bc;
		return mutant; 
	}
	
	public TestCase<BankChurnersValue, Integer> realFeatureDown (int feature, TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = new TestCase<BankChurnersValue, Integer>();
		BankChurnersValue bc = new BankChurnersValue();
		for (int i=0; i<=2; i++) {
			bc.realValue[i] = seed.input.realValue[i];
		};
		bc.realValue[feature] -= BankChurnersFeatures.realStep[feature];
		if (bc.realValue[feature] < BankChurnersFeatures.realLower[feature]) {
			bc.realValue[feature] = BankChurnersFeatures.realLower[feature];
		}
		mutant.input = bc;
		return mutant; 
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Gender_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = discFeatureUp(BankChurnersFeatures.Gender, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Gender_Down (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = discFeatureDown(BankChurnersFeatures.Gender, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Education_Level_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = discFeatureUp(BankChurnersFeatures.Education_Level, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Education_Level_Down (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = discFeatureDown(BankChurnersFeatures.Education_Level, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Marital_Status_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = discFeatureUp(BankChurnersFeatures.Marital_Status, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Marital_Status_Down (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = discFeatureDown(BankChurnersFeatures.Marital_Status, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Income_Category_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = discFeatureUp(BankChurnersFeatures.Income_Category, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Income_Category_Down (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = discFeatureDown(BankChurnersFeatures.Income_Category, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Card_Category_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = discFeatureUp(BankChurnersFeatures.Card_Category, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Card_Category_Down (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = discFeatureDown(BankChurnersFeatures.Card_Category, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Customer_Age_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureUp(BankChurnersFeatures.Customer_Age, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Customer_Age_Down (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureDown(BankChurnersFeatures.Customer_Age, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Dependent_count_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureUp(BankChurnersFeatures.Dependent_count, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Dependent_count_Down (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureDown(BankChurnersFeatures.Dependent_count, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Months_on_book_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureUp(BankChurnersFeatures.Months_on_book, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Months_on_book_Down (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureDown(BankChurnersFeatures.Months_on_book, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Total_Relationship_Count_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureUp(BankChurnersFeatures.Total_Relationship_Count, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Total_Relationship_Count_Down (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureDown(BankChurnersFeatures.Total_Relationship_Count, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Months_Inactive_12_mon_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureUp(BankChurnersFeatures.Months_Inactive_12_mon, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Months_Inactive_12_mon_Down (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureDown(BankChurnersFeatures.Months_Inactive_12_mon, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Contacts_Count_12_mon_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureUp(BankChurnersFeatures.Contacts_Count_12_mon, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Contacts_Count_12_mon_Down (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureDown(BankChurnersFeatures.Contacts_Count_12_mon, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Credit_Limit_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureUp(BankChurnersFeatures.Credit_Limit, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Credit_Limit_Down (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureDown(BankChurnersFeatures.Credit_Limit, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Total_Revolving_Bal_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureUp(BankChurnersFeatures.Total_Revolving_Bal, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Total_Revolving_Bal_Down (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureDown(BankChurnersFeatures.Total_Revolving_Bal, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Avg_Open_To_Buy_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureUp(BankChurnersFeatures.Avg_Open_To_Buy, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Avg_Open_To_Buy_Down (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureDown(BankChurnersFeatures.Avg_Open_To_Buy, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Total_Trans_Amt_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureUp(BankChurnersFeatures.Total_Trans_Amt, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Total_Trans_Amt_Down (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureDown(BankChurnersFeatures.Total_Trans_Amt, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Total_Trans_Ct_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureUp(BankChurnersFeatures.Total_Trans_Ct, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Total_Trans_Ct_Down (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = intFeatureDown(BankChurnersFeatures.Total_Trans_Ct, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Total_Amt_Chng_Q4_Q1_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = realFeatureUp(BankChurnersFeatures.Total_Amt_Chng_Q4_Q1, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Total_Amt_Chng_Q4_Q1_Down (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = realFeatureDown(BankChurnersFeatures.Total_Amt_Chng_Q4_Q1, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Total_Ct_Chng_Q4_Q1_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = realFeatureUp(BankChurnersFeatures.Total_Ct_Chng_Q4_Q1, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Avg_Utilization_Ratio_Down (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = realFeatureDown(BankChurnersFeatures.Avg_Utilization_Ratio, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> Avg_Utilization_Ratio_Up (TestCase<BankChurnersValue,Integer> seed){
		TestCase<BankChurnersValue, Integer> mutant = realFeatureUp(BankChurnersFeatures.Avg_Utilization_Ratio, seed);
		return mutant ;
	}
	
	@Datamorphism
	public TestCase<BankChurnersValue, Integer> mid (
				TestCase<BankChurnersValue,Integer> seed1, 
				TestCase<BankChurnersValue,Integer> seed2){
		TestCase<BankChurnersValue, Integer> mutant = new TestCase<BankChurnersValue, Integer>();
		BankChurnersValue bc = new BankChurnersValue();
		boolean toChange = true; 
		for (int i=0; i<=4; i++) {
			bc.discValue[i] = seed1.input.discValue[i];
			if (seed1.input.discValue[i] != seed2.input.discValue[i]) {
				if (toChange) {
					bc.discValue[i] = seed2.input.discValue[i];
				}
				toChange = !toChange; 
			}
		}
		for (int i=0; i<=10; i++) {
			bc.intValue[i] = (seed1.input.intValue[i] + seed2.input.intValue[i])/2;
		}
		for (int i=0; i<=2; i++) {
			bc.realValue[i] = (seed1.input.realValue[i] + seed2.input.realValue[i])/2;
		}
		mutant.input = bc;
		return mutant ;
	}

	@Override
	public double distance(TestCase<BankChurnersValue, Integer> x, TestCase<BankChurnersValue, Integer> y) {
		double dist = 0; 
		for (int i = 0; i<5; i++) {
			if (x.input.discValue[i] != y.input.discValue[i]) {dist++;}
		};
		for (int i=0; i<11; i++) {
			dist += Math.abs(x.input.intValue[i] - y.input.intValue[i])
					/(BankChurnersFeatures.intUpper[i]-BankChurnersFeatures.intLower[i]);
		};
		for (int i=0; i<3; i++) {
			dist += Math.abs(x.input.realValue[i] - y.input.realValue[i])
					/(BankChurnersFeatures.realUpper[i]-BankChurnersFeatures.realLower[i]);
		}
//		System.out.println("Distance = "+ dist);
		return dist;
	}
}
