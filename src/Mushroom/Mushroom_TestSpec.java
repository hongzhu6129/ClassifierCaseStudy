package Mushroom;

import java.io.File;
import java.io.IOException;

import javax.swing.JFileChooser;
import javax.swing.filechooser.FileFilter;

import morphy.annotations.*;  

public class Mushroom_TestSpec extends MushroomMorphisms {
	public ModelInvoker invoker;
	public String pyScriptFileName = "";
	
	@TestExecuter
	public Integer execute(MushroomValue x) {
		String arguements = "";
		for (int i=0; i<22; i++) {
			arguements = arguements + "," + x.value[i];
		}
		arguements = arguements.substring(1); 
		try { 
			String result = invoker.invokeModel(arguements);
			System.out.print("."); 
			int mushroomClass = Integer.valueOf(result);
			return mushroomClass;
		} catch (Exception e) {
			System.out.println("Failed to invoke Python script on parameters: " + arguements);
//			e.printStackTrace();
			return -1;
		}
	}
	
	@Analyser
	public void start_Invoker() {
		JFileChooser fileChooser = new JFileChooser("C:\\Morphy\\Mushroom\\PyScripts");
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
	
	public TestPool<MushroomValue, Integer> expected = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> DT = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> DT2 = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> HV = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> HV2 = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> KNN = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> KNN2 = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> LR = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> LR2 = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> NB = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> NB2 = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> RF = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> RF2 = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> Stack = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> Stack2 = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> Stack3 = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> SV = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> SV2 = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> SVM = new TestPool<MushroomValue, Integer>();
	
	public TestPool<MushroomValue, Integer> SVM2 = new TestPool<MushroomValue, Integer>();
	
	@Analyser
	public void save_to_Expected() {
		expected = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			expected.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_DT() {
		DT = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			DT.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_DT2() {
		DT2 = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			DT2.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_HV() {
		HV = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			HV.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_HV2() {
		HV2 = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			HV2.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_KNN() {
		KNN = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			KNN.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_KNN2() {
		KNN2 = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			KNN2.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_LR() {
		LR = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			LR.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_LR2() {
		LR2 = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			LR2.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_NB() {
		NB = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			NB.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_NB2() {
		NB2 = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			NB2.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_RF() {
		RF = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			RF.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_Stack() {
		Stack = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			Stack.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_Stack2() {
		Stack2 = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			Stack2.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_Stack3() {
		Stack3 = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			Stack3.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_SV() {
		SV = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			SV.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_SV2() {
		SV2 = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			SV2.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_SVM() {
		SVM = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			SVM.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_SVM2() {
		SVM2 = new TestPool<MushroomValue, Integer>();
		for (TestCase<MushroomValue, Integer> x: testSuite.testSet) {
			SVM2.addTestCase(x);
		}
	}
	
	@Metamorphism(message="Does not equal to the expected value.")
	public boolean equ_Expected(TestCase<MushroomValue, Integer> x) {
		return (expected.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to DT.")
	public boolean equ_DT(TestCase<MushroomValue, Integer> x) {
		return (DT.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to DT2.")
	public boolean equ_DT2(TestCase<MushroomValue, Integer> x) {
		return (DT2.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to HV.")
	public boolean equ_HV(TestCase<MushroomValue, Integer> x) {
		return (HV.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to HV2.")
	public boolean equ_HV2(TestCase<MushroomValue, Integer> x) {
		return (HV2.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to KNN.")
	public boolean equ_KNN(TestCase<MushroomValue, Integer> x) {
		return (KNN.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to KNN2.")
	public boolean equ_KNN2(TestCase<MushroomValue, Integer> x) {
		return (KNN2.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to LR.")
	public boolean equ_LR(TestCase<MushroomValue, Integer> x) {
		return (LR.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to LR2.")
	public boolean equ_LR2(TestCase<MushroomValue, Integer> x) {
		return (LR2.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to NB.")
	public boolean equ_NB(TestCase<MushroomValue, Integer> x) {
		return (NB.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to NB2.")
	public boolean equ_NB2(TestCase<MushroomValue, Integer> x) {
		return (NB2.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to RF.")
	public boolean equ_RF(TestCase<MushroomValue, Integer> x) {
		return (RF.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to RF2.")
	public boolean equ_RF2(TestCase<MushroomValue, Integer> x) {
		return (RF2.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to Stack.")
	public boolean equ_Stack(TestCase<MushroomValue, Integer> x) {
		return (Stack.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to Stack2.")
	public boolean equ_Stack2(TestCase<MushroomValue, Integer> x) {
		return (Stack2.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to Stack3.")
	public boolean equ_Stack3(TestCase<MushroomValue, Integer> x) {
		return (Stack3.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to SV.")
	public boolean equ_SV(TestCase<MushroomValue, Integer> x) {
		return (SV.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to SV2.")
	public boolean equ_SV2(TestCase<MushroomValue, Integer> x) {
		return (SV2.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to SVM.")
	public boolean equ_SVM(TestCase<MushroomValue, Integer> x) {
		return (SVM.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to SVM2.")
	public boolean equ_SVM2(TestCase<MushroomValue, Integer> x) {
		return (SVM2.get(x.id).output == x.output);
	}
}
