package BankChurners;

import java.io.File;
import java.io.IOException;

import javax.swing.JFileChooser;
import javax.swing.filechooser.FileFilter;

import morphy.annotations.*;  

public class BankChurners_TestSpec extends BankChurnersMorphisms {
	public ModelInvoker invoker;
	public String pyScriptFileName = "";
	
	@TestExecuter
	public Integer execute(BankChurnersValue x) {
		String arguements = "";
		for (int i=0; i<=4; i++) {
			arguements += "," + x.discValue[i];
		}
		for (int i=0; i<=10; i++) {
			arguements += "," + x.intValue[i];
		}
		for (int i=0; i<=2; i++) {
			arguements += "," + x.realValue[i];
		}
		arguements = arguements.substring(1); 
		try { 
			String result = invoker.invokeModel(arguements);
			System.out.print("."); 
			int customerAttrite = Integer.valueOf(result);
			return customerAttrite;
		} catch (Exception e) {
			System.out.println("Failed to invoke Python script on parameters: " + arguements);
//			e.printStackTrace();
			return -1;
		}
	}
	
	@Analyser
	public void start_Invoker() {
		JFileChooser fileChooser = new JFileChooser("C:\\Morphy\\BankChurners\\PyScripts");
		fileChooser.setFileSelectionMode(JFileChooser.FILES_ONLY);
		PyFileFilter filter = new PyFileFilter();
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
	
	public TestPool<BankChurnersValue, Integer> expected = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> DT = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> DT2 = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> HV = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> HV2 = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> KNN = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> KNN2 = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> LR = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> LR2 = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> NB = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> NB2 = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> RF = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> RF2 = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> Stack = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> Stack2 = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> Stack3 = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> SV = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> SV2 = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> SVM = new TestPool<BankChurnersValue, Integer>();
	
	public TestPool<BankChurnersValue, Integer> SVM2 = new TestPool<BankChurnersValue, Integer>();
	
	public void copyTestCase(TestCase<BankChurnersValue, Integer> x, 
			TestCase<BankChurnersValue, Integer> y) {
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
		expected = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			expected.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_DT() {
		DT = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			DT.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_DT2() {
		DT2 = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			DT2.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_HV() {
		HV = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			HV.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_HV2() {
		HV2 = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			HV2.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_KNN() {
		KNN = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			KNN.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_KNN2() {
		KNN2 = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			KNN2.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_LR() {
		LR = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			LR.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_LR2() {
		LR2 = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			LR2.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_NB() {
		NB = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			NB.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_NB2() {
		NB2 = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			NB2.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_RF() {
		RF = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			RF.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_Stack() {
		Stack = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			Stack.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_Stack2() {
		Stack2 = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			Stack2.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_Stack3() {
		Stack3 = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			Stack3.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_SV() {
		SV = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			SV.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_SV2() {
		SV2 = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			SV2.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_SVM() {
		SVM = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			SVM.addTestCase(y);
		}
	}
	
	@Analyser
	public void save_to_SVM2() {
		SVM2 = new TestPool<BankChurnersValue, Integer>();
		for (TestCase<BankChurnersValue, Integer> x: testSuite.testSet) {
			TestCase<BankChurnersValue, Integer> y = new TestCase();
			copyTestCase(x,y);
			SVM2.addTestCase(y);
		}
	}
	
	@Metamorphism(message="Does not equal to the expected value.")
	public boolean equ_Expected(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = expected.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to DT.")
	public boolean equ_DT(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = DT.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to DT2.")
	public boolean equ_DT2(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = DT2.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to HV.")
	public boolean equ_HV(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = HV.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to HV2.")
	public boolean equ_HV2(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = HV2.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to KNN.")
	public boolean equ_KNN(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = KNN.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to KNN2.")
	public boolean equ_KNN2(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = KNN2.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to LR.")
	public boolean equ_LR(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = LR.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to LR2.")
	public boolean equ_LR2(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = LR2.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to NB.")
	public boolean equ_NB(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = NB.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to NB2.")
	public boolean equ_NB2(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = NB2.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to RF.")
	public boolean equ_RF(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = RF.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to RF2.")
	public boolean equ_RF2(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = RF2.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to Stack.")
	public boolean equ_Stack(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = Stack.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to Stack2.")
	public boolean equ_Stack2(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = Stack2.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to Stack3.")
	public boolean equ_Stack3(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = Stack3.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to SV.")
	public boolean equ_SV(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = SV.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to SV2.")
	public boolean equ_SV2(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = SV2.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to SVM.")
	public boolean equ_SVM(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = SVM.get(x.id);
		return (y.output == x.output);
	}
	
	@Metamorphism(message="Does not equal to SVM2.")
	public boolean equ_SVM2(TestCase<BankChurnersValue, Integer> x) {
		TestCase<BankChurnersValue, Integer> y = SVM2.get(x.id);
		return (y.output == x.output);
	}
}
