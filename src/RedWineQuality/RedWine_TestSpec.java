package RedWineQuality;

import java.io.File;
import java.io.IOException;

import javax.swing.JFileChooser;
import javax.swing.filechooser.FileFilter;

import morphy.annotations.*;  

public class RedWine_TestSpec extends RedWineMorphisms {
	public ModelInvoker invoker;
	public String pyScriptFileName = "";
	
	@TestExecuter
	public Integer execute(RedWineFeatures x) {
		String arguements = x.fixedAcidity +"," 
				+ x.volatileAcidity +"," 
				+ x.citricAcid + ","
				+ x.residualSugar + ","
				+ x.chlorides +","
				+ x.freeSulfurDioxide +","
				+ x.totalSulfurDioxide + ","
				+ x.density + ","
				+ x.pH +","
				+ x.sulphates + ","
				+ x.alcohol;
		try { 
			String result = invoker.invokeModel(arguements);
			System.out.print("."); 
			int quality = Integer.valueOf(result);
			return quality;
		} catch (Exception e) {
			System.out.println("Failed to invoke Python script on parameters: " + arguements);
//			e.printStackTrace();
			return -1;
		}
	}
	
	@Analyser
	public void start_Invoker() {
		JFileChooser fileChooser = new JFileChooser("C:\\Morphy\\RedWineQuality\\PyScripts");
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
	
	public TestPool<RedWineFeatures, Integer> expected = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> DT = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> DT2 = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> HV = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> HV2 = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> KNN = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> KNN2 = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> LR = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> LR2 = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> NB = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> NB2 = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> RF = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> RF2 = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> Stack = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> Stack2 = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> Stack3 = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> SV = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> SV2 = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> SVM = new TestPool<RedWineFeatures, Integer>();
	
	public TestPool<RedWineFeatures, Integer> SVM2 = new TestPool<RedWineFeatures, Integer>();
	
	@Analyser
	public void save_to_Expected() {
		expected = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			expected.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_DT() {
		DT = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			DT.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_DT2() {
		DT2 = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			DT2.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_HV() {
		HV = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			HV.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_HV2() {
		HV2 = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			HV2.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_KNN() {
		KNN = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			KNN.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_KNN2() {
		KNN2 = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			KNN2.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_LR() {
		LR = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			LR.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_LR2() {
		LR2 = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			LR2.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_NB() {
		NB = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			NB.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_NB2() {
		NB2 = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			NB2.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_RF() {
		RF = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			RF.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_Stack() {
		Stack = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			Stack.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_Stack2() {
		Stack2 = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			Stack2.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_Stack3() {
		Stack3 = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			Stack3.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_SV() {
		SV = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			SV.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_SV2() {
		SV2 = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			SV2.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_SVM() {
		SVM = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			SVM.addTestCase(x);
		}
	}
	
	@Analyser
	public void save_to_SVM2() {
		SVM2 = new TestPool<RedWineFeatures, Integer>();
		for (TestCase<RedWineFeatures, Integer> x: testSuite.testSet) {
			SVM2.addTestCase(x);
		}
	}
	
	@Metamorphism(message="Does not equal to the expected value.")
	public boolean equ_Expected(TestCase<RedWineFeatures, Integer> x) {
		return (expected.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to DT.")
	public boolean equ_DT(TestCase<RedWineFeatures, Integer> x) {
		return (DT.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to DT2.")
	public boolean equ_DT2(TestCase<RedWineFeatures, Integer> x) {
		return (DT2.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to HV.")
	public boolean equ_HV(TestCase<RedWineFeatures, Integer> x) {
		return (HV.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to HV2.")
	public boolean equ_HV2(TestCase<RedWineFeatures, Integer> x) {
		return (HV2.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to KNN.")
	public boolean equ_KNN(TestCase<RedWineFeatures, Integer> x) {
		return (KNN.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to KNN2.")
	public boolean equ_KNN2(TestCase<RedWineFeatures, Integer> x) {
		return (KNN2.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to LR.")
	public boolean equ_LR(TestCase<RedWineFeatures, Integer> x) {
		return (LR.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to LR2.")
	public boolean equ_LR2(TestCase<RedWineFeatures, Integer> x) {
		return (LR2.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to NB.")
	public boolean equ_NB(TestCase<RedWineFeatures, Integer> x) {
		return (NB.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to NB2.")
	public boolean equ_NB2(TestCase<RedWineFeatures, Integer> x) {
		return (NB2.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to RF.")
	public boolean equ_RF(TestCase<RedWineFeatures, Integer> x) {
		return (RF.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to RF2.")
	public boolean equ_RF2(TestCase<RedWineFeatures, Integer> x) {
		return (RF2.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to Stack.")
	public boolean equ_Stack(TestCase<RedWineFeatures, Integer> x) {
		return (Stack.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to Stack2.")
	public boolean equ_Stack2(TestCase<RedWineFeatures, Integer> x) {
		return (Stack2.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to Stack3.")
	public boolean equ_Stack3(TestCase<RedWineFeatures, Integer> x) {
		return (Stack3.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to SV.")
	public boolean equ_SV(TestCase<RedWineFeatures, Integer> x) {
		return (SV.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to SV2.")
	public boolean equ_SV2(TestCase<RedWineFeatures, Integer> x) {
		return (SV2.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to SVM.")
	public boolean equ_SVM(TestCase<RedWineFeatures, Integer> x) {
		return (SVM.get(x.id).output == x.output);
	}
	
	@Metamorphism(message="Does not equal to SVM2.")
	public boolean equ_SVM2(TestCase<RedWineFeatures, Integer> x) {
		return (SVM2.get(x.id).output == x.output);
	}
}
