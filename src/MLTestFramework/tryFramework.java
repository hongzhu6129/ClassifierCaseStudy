package MLTestFramework;

import java.lang.reflect.Field;

import RedWineQuality.RedWineFeatures;
import morphy.annotations.*;

public class tryFramework extends MLTest<Integer, Integer> {
	@TestSetContainer(
			inputTypeName = "Integer",
			outputTypeName = "Integer")
	public TestPool<Integer, Integer> testSuite2 = testSuite;
	
	@Override
	public double distance(TestCase<Integer, Integer> x, TestCase<Integer, Integer> y) {
		return Math.abs(x.input -y.input);
	}

	@TestExecuter
	public Integer execute(int x) {
			return x;
	}
}
