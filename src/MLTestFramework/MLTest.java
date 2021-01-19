package MLTestFramework;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;

import javax.swing.JFileChooser;
import javax.swing.JOptionPane;

import morphy.annotations.Analyser;
import morphy.annotations.Metamorphism;
import morphy.annotations.TestCase;
import morphy.annotations.TestDataFeature;
import morphy.annotations.TestExecuter;
import morphy.annotations.TestPool;
import morphy.annotations.TestSetContainer;

public abstract class MLTest<inType, outType> {

	public TestPool<inType, outType> testSuite = new TestPool<inType, outType>();
		
	public ModelInvoker invoker;
	public PyFileFilter filter;
	public String pyScriptFileName = "";
	
	@Analyser
	public void start_Invoker() {
		JFileChooser fileChooser = new JFileChooser("C:\\Morphy\\BankChurners\\PyScripts");
		fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
		filter = new PyFileFilter();
		fileChooser.addChoosableFileFilter(filter);
		fileChooser.showOpenDialog(null);
		pyScriptFileName = fileChooser.getSelectedFile().getName();
		String pyScriptFileNameFull = fileChooser.getSelectedFile().getAbsolutePath();
		try {
			System.out.println("Selected file="+ pyScriptFileNameFull);
			invoker = new ModelInvoker(pyScriptFileNameFull);
		} catch (IOException e) {
			System.out.println("Failed to start Python script "+ pyScriptFileName);
//			e.printStackTrace();
		}
	}
	
	@Analyser
	public void stop_Invoker() {
		try {
			String result = invoker.invokeModel("");
			System.out.println(result + pyScriptFileName);
		} catch (IOException e) {
			System.out.println("Failed to stop Python script "+ pyScriptFileName);
//			e.printStackTrace();
		}
	}
	
	public TestPool<inType, outType> expected = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> DT = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> DT2 = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> HV = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> HV2 = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> KNN = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> KNN2 = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> LR = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> LR2 = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> NB = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> NB2 = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> RF = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> RF2 = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> Stack = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> Stack2 = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> Stack3 = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> SV = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> SV2 = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> SVM = new TestPool<inType, outType>();
	
	public TestPool<inType, outType> SVM2 = new TestPool<inType, outType>();
	
	public void copyTestCase(TestCase<inType, outType> x, TestCase<inType, outType> y) {
		y.id = x.id;
		y.input = x.input;
		y.output = x.output;
		y.feature = x.feature;
		y.origins = x.origins;
		y.correctness = x.correctness;
		y.setType(x.getType());
	}
	
	@Analyser
	public void save_to_Expected() {
		expected = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			expected.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_DT() {
		DT = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			DT.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_DT2() {
		DT2 = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			DT2.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_HV() {
		HV = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			HV.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_HV2() {
		HV2 = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			HV2.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_KNN() {
		KNN = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			KNN.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_KNN2() {
		KNN2 = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			KNN2.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_LR() {
		LR = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			LR.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_LR2() {
		LR2 = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			LR2.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_NB() {
		NB = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			NB.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_NB2() {
		NB2 = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			NB2.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_RF() {
		RF = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			RF.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_Stack() {
		Stack = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			Stack.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_Stack2() {
		Stack2 = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			Stack2.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_Stack3() {
		Stack3 = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			Stack3.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_SV() {
		SV = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			SV.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_SV2() {
		SV2 = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			SV2.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_SVM() {
		SVM = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			SVM.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_SVM2() {
		SVM2 = new TestPool<inType, outType>();
		for (TestCase<inType, outType> x: testSuite.testSet) {
			TestCase<inType, outType> y = new TestCase();
			copyTestCase(x,y);
			SVM2.addTestCase(y);
		}
	}
	
	@Metamorphism(message="Does not equal to the expected value.")
	public boolean equ_Expected(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = expected.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to DT.")
	public boolean equ_DT(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = DT.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to DT2.")
	public boolean equ_DT2(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = DT2.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to HV.")
	public boolean equ_HV(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = HV.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to HV2.")
	public boolean equ_HV2(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = HV2.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to KNN.")
	public boolean equ_KNN(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = KNN.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to KNN2.")
	public boolean equ_KNN2(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = KNN2.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to LR.")
	public boolean equ_LR(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = LR.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to LR2.")
	public boolean equ_LR2(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = LR2.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to NB.")
	public boolean equ_NB(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = NB.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to NB2.")
	public boolean equ_NB2(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = NB2.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to RF.")
	public boolean equ_RF(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = RF.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to RF2.")
	public boolean equ_RF2(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = RF2.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to Stack.")
	public boolean equ_Stack(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = Stack.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to Stack2.")
	public boolean equ_Stack2(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = Stack2.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to Stack3.")
	public boolean equ_Stack3(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = Stack3.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to SV.")
	public boolean equ_SV(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = SV.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to SV2.")
	public boolean equ_SV2(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = SV2.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to SVM.")
	public boolean equ_SVM(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = SVM.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to SVM2.")
	public boolean equ_SVM2(TestCase<inType, outType> x) {
		TestCase<inType, outType> y = SVM2.get(x.id);
		return (y.output == x.output);
	}
	
	@Analyser
	public void statisticsOfCorrectness() {
		int numTC = testSuite.testSet.size();
		int numOriginalTC = 0;
		int numMutantTC = 0;
		int numCheckedTC =0;
		int numCorrect =0;
		int numError=0;
		int numFailedTestCases = 0; 
		HashMap<String, Integer> correctnessMap = new HashMap<String, Integer>();
		for (TestCase x : testSuite.testSet) {
			if (x.feature == TestDataFeature.original) { 
				numOriginalTC++;
			}else {
				numMutantTC++;
			};
			if (!x.correctness.equals(null)) {
				String correctness = x.correctness;
				if (correctness.equals("")) continue; 
				numCheckedTC++;
				if (correctness.contains("fail")) {
					numFailedTestCases ++;
				}
				String[] correctnessRecords = correctness.split(";");
				for (String record: correctnessRecords) {
					String key = record.replace("=","<")+">";
					if (correctnessMap.keySet().contains(key) ) {
						int num = correctnessMap.get(key);
						num++;
						correctnessMap.put(key, num);
					}else {
						correctnessMap.put(key, 1);
					}
				}
			}
		};
		String message = "Statistics:\n"; 
		message = message + "Total number of test cases = " + numTC + "\n"; 
		message = message + "Number of original test cases = "+ numOriginalTC + "\n";
		message = message + "Number of mutant test cases = " + numMutantTC + "\n";
		message = message + "Number of test cases checked = " + numCheckedTC + "\n";
		message = message + "Number of test cases failed checking = " + numFailedTestCases + "\n";
		message = message + "Failure Rate = " + ((double)numFailedTestCases/(double)numCheckedTC)*100 + "%\n";
		List<String> keysSorted = new ArrayList<String>();
		for (String key : correctnessMap.keySet()){
			keysSorted.add(key);
		};
		Collections.sort(keysSorted);
		for (String key: keysSorted) {
			message = message + " -- number of " + key + " = " + correctnessMap.get(key) +"\n";
		}
		for (String key : keysSorted) {
			if (key.contains("pass")) {
				numCorrect = numCorrect+correctnessMap.get(key);
			}else {
				numError = numError + correctnessMap.get(key);
			}
		}
		message = message + "Number of times checking passed = " + numCorrect + "\n";
		message = message + "Number of times checking failed = " + numError + "\n";
		message = message + "Failure rate = " + ((double) (numError) / (double) ((numCorrect + numError)))*100 + "%\n";
		JOptionPane.showMessageDialog(null, message);
	}
	
	@Analyser
	public void StatisticsOfDirecetedWalk() {
		//Open file where results of directed walks are stored:
		String dir = System.getProperty("user.dir");
		String fileName = dir + File.separator + "res.txt";
		File resFile = new File(fileName);
		FileReader fileReader;
		try {
			fileReader = new FileReader(resFile);
		} catch (FileNotFoundException e1) {
			System.out.println("File not find: " + fileName);
			return;
		};
		BufferedReader reader = new BufferedReader(fileReader);
		
		//Read data from file and sort the data according to walking method: 
		String line;
		HashMap<String, List<String>> ResultMapDW = new HashMap<String, List<String>>();
		try {
			while ((line = reader.readLine()) != null) {
				if (line.startsWith("Directed Walk")) {
					String[] resultDetail = line.split(",");
					String record = resultDetail[3].substring(12) + ","
							+ resultDetail[4].substring(13) + ","
							+ resultDetail[5].substring(13);
					String key = resultDetail[1].substring(12);
					if (ResultMapDW.keySet().contains(key) ) {
						List<String> records = ResultMapDW.get(key);
						records.add(record);
						ResultMapDW.put(key, records);
					}else {
						List<String> records = new ArrayList<String>();
						records.add(record);
						ResultMapDW.put(key, records);
					}
				}
			}
		}catch(IOException e) {
			System.out.println("File read error: " + fileName);
			return;
		}
		try {
			reader.close();
		} catch (IOException e) {
			//	e.printStackTrace();
			System.out.println("File failed to close: " + fileName);
			return;
		}
		
		// Statistics of the distances from the start point to border: 
		HashMap<String, Double> statMapMax = new HashMap<String, Double>();
		HashMap<String, Double> statMapMin = new HashMap<String, Double>();
		HashMap<String, Double> statMapSum = new HashMap<String, Double>();
		HashMap<String, Integer> statMapCnt = new HashMap<String, Integer>();
		String message = ""; 
		for (String key : ResultMapDW.keySet()){
			for (String record : ResultMapDW.get(key)) {
					String[] points = new String[3];
					points = record.split(",");
					TestCase<inType, outType> startPoint = testSuite.get(points[0]);
					if (startPoint == null) {continue;}
					TestCase<inType, outType> pointA = testSuite.get(points[1]);
					TestCase<inType, outType> pointB = testSuite.get(points[2]);
					double dist = 0; 
					if ((pointA != null) && (pointA.output == startPoint.output)) {
						 dist = distance(startPoint, pointA);
					};
					if ((pointB !=null) && (pointB.output == startPoint.output)) {
						dist = distance(startPoint, pointB);
					}
					if (dist > 0) {
						if (statMapCnt.containsKey(key)) {
							int cnt = statMapCnt.get(key) +1;
							statMapCnt.put(key,cnt);
							double max = statMapMax.get(key);
							if (dist > max) { statMapMax.put(key, dist);}
							double min = statMapMin.get(key);
							if (dist < min) { statMapMin.put(key, dist);}
							double sum = statMapSum.get(key);
							statMapSum.put(key, sum+dist);
						}else {
							statMapCnt.put(key,1);
							statMapMax.put(key, dist);
							statMapMin.put(key, dist);
							statMapSum.put(key, dist);
						}
					}
			};
			message += "Direction:" + key 
				+ ", Count: " + statMapCnt.get(key)
				+ ", Max: " + String.format("%.04f", statMapMax.get(key))
				+ ", Min: " + String.format("%.04f", statMapMin.get(key))
				+ ", Average: " + String.format("%.04f", statMapSum.get(key)/statMapCnt.get(key)) + "\n";
		}
		
		// display results
		message = "Statistic Result of Directed walk Exploration\n" + message;
		JOptionPane.showMessageDialog(null, message);
	}
	
	@Analyser
	public void StatisticsOfRandomWalk() {
		//Open file where results of directed walks are stored:
		String dir = System.getProperty("user.dir");
		String fileName = dir + File.separator + "res.txt";
		File resFile = new File(fileName);
		FileReader fileReader;
		try {
			fileReader = new FileReader(resFile);
		} catch (FileNotFoundException e1) {
			System.out.println("File not find: " + fileName);
			return;
		};
		BufferedReader reader = new BufferedReader(fileReader);
		
		//Read data from file and sort the data according to walking method: 
		String line;
		HashMap<String, List<String>> ResultMapDW = new HashMap<String, List<String>>();
		try {
			while ((line = reader.readLine()) != null) {
				if (line.startsWith("Random Walk")) {
					String[] resultDetail = line.split(",");
					String record = // resultDetail[2].substring(12) + "," +
							 resultDetail[3].substring(13) + ","
							+ resultDetail[4].substring(13);
					String key = resultDetail[2].substring(12);
					if (ResultMapDW.keySet().contains(key) ) {
						List<String> records = ResultMapDW.get(key);
						records.add(record);
						ResultMapDW.put(key, records);
					}else {
						List<String> records = new ArrayList<String>();
						records.add(record);
						ResultMapDW.put(key, records);
					}
				}
			}
		}catch(IOException e) {
			System.out.println("File read error: " + fileName);
			return;
		}
		try {
			reader.close();
		} catch (IOException e) {
			//	e.printStackTrace();
			System.out.println("File failed to close: " + fileName);
			return;
		}
		
		// Statistics of the distances from the start point to border: 
		HashMap<String, Double> statMapMax = new HashMap<String, Double>();
		HashMap<String, Double> statMapMin = new HashMap<String, Double>();
		HashMap<String, Double> statMapSum = new HashMap<String, Double>();
		HashMap<String, Integer> statMapCnt = new HashMap<String, Integer>();
		String message = ""; 
		double overallMax = 0;
		double overallMin = -1;
		double overallCnt = 0;
		double overallSum = 0;
		for (String key : ResultMapDW.keySet()){
//			System.out.println("Process key: "+ key);
			for (String record : ResultMapDW.get(key)) {
					TestCase<inType, outType> startPoint = testSuite.get(key);
					if (startPoint == null) {continue;}
					String[] points = new String[2];
					points = record.split(",");
//					System.out.println("Process point A: "+ points[0]);
					TestCase<inType, outType> pointA = testSuite.get(points[0]);
//					System.out.println("Process point B: "+ points[1]);
					TestCase<inType, outType> pointB = testSuite.get(points[1]);
					double dist = 0; 
					if ((pointA != null) && (pointA.output == startPoint.output)) {
						 dist = distance(startPoint, pointA);
					};
					if ((pointB !=null) && (pointB.output == startPoint.output)) {
						dist = distance(startPoint, pointB);
					}
					if (dist > 0) {
						if (statMapCnt.containsKey(key)) {
							int cnt = statMapCnt.get(key) +1;
							statMapCnt.put(key,cnt);
							double max = statMapMax.get(key);
							if (dist > max) { statMapMax.put(key, dist);}
							double min = statMapMin.get(key);
							if (dist < min) { statMapMin.put(key, dist);}
							double sum = statMapSum.get(key);
							statMapSum.put(key, sum+dist);
						}else {
							statMapCnt.put(key,1);
							statMapMax.put(key, dist);
							statMapMin.put(key, dist);
							statMapSum.put(key, dist);
						}
					}
			};
			if (statMapCnt.containsKey(key)) {
				double keyMax = statMapMax.get(key);
				double keyMin = statMapMin.get(key);
				int keyCnt = statMapCnt.get(key);
				double keySum = statMapSum.get(key);
				message += "Start Point: " + key 
						+ ", Count: " + statMapCnt.get(key)
						+ ", Max: " + String.format("%.02f", keyMax)
						+ ", Min: " + String.format("%.02f", keyMin)
						+ ", Average: " + String.format("%.02f", keySum/keyCnt) + "\n";
			
				if (keyMax > overallMax) {overallMax = keyMax;}
				if (overallMin<0 || keyMin < overallMin) {overallMin = keyMin;}
				overallCnt += keyCnt;
				overallSum += keySum;
			}
		}
		
		// display results
		message += "----------------------------------\n";
		message += "Overall Cnt: " + overallCnt
			+ ", Overall Max: " + String.format("%.02f", overallMax) 
			+ ", Overall Min: " + String.format("%.02f", overallMin)
			+ ", Overall Avg: " + String.format("%.02f", overallSum/overallCnt);
		message = "Statistic Result of Random Walk Exploration:\n" + message;
		JOptionPane.showMessageDialog(null, message);
	}
	
	public abstract double distance (TestCase<inType, outType> x, TestCase<inType, outType> y);
}
