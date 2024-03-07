
/*
原文: https:mp.weixin.qq.com/s/-rAosHEpBytJ8wvEmVSFtw, 部分代码做了调整, 尤其是5
Java Stream API对于 Java 开发人员来说就像一把瑞士军刀 — 它用途广泛、结构紧凑，并且可以轻松处理各种任务。
它为开发人员提供了一种函数式和声明式的方式来表达复杂的数据转换和操作，使代码更加简洁和富有表现力。
但能力越大，责任越大，有效地使用Stream API需要对最佳实践和常见陷阱有深入的了解。
今天，我们将探讨使用Java Stream API的一些最佳实践，并展示如何释放这个神奇工具的全部潜力。
*/

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class StreamSkills {
    /*
    1.使用原始流以获得更好的性能
    使用 int、long 和 double 等基本类型时，请使用IntStream、LongStream 和 DoubleStream 等基本流，
    而不是 Integer、Long 和 Double 等装箱类型流。原始流可以通过避免装箱和拆箱的成本来提供更好的性能。
    */
    private static void test1() {
        var array = new int[]{1, 2, 3, 4, 5};
        var sum = Arrays.stream(array)
                .sum();
        System.out.println(sum);
    }

    /*
    2. 避免嵌套流
    最佳实践是避免嵌套流，因为它可能导致代码难以阅读和理解。相反，尝试将问题分解为更小的部分，并使用中间集合或局部变量来存储中间结果。
    */
    private static void test2() {
        var list1 = Arrays.asList("apple", "banana", "cherry");
        var list2 = Arrays.asList("orange", "pineapple", "mango");
        var result = Stream.concat(list1.stream(), list2.stream())
                .filter(s -> s.length() > 5)
                .collect(Collectors.toList());
        System.out.println(result);
    }

    /*
    3. 谨慎使用并行流
    并行流可以在处理大量数据时提供更好的性能，但它们也会引入开销和竞争条件。谨慎使用并行流，并考虑数据大小、
    操作复杂性和可用处理器数量等因素。
    */
    private static void test3() {
        var list = Arrays.asList(1, 2, 3, 4, 5);
        var sum = list.parallelStream().reduce(0, Integer::sum);
        System.out.println(sum);
    }

    /*
    4. 使用惰性求值以获得更好的性能
    Stream API 支持延迟计算，这意味着在调用终端操作之前不会执行中间操作。作为最佳实践，尝试使用惰性计算来通过减少不必要的计算来提高性能。
    */
    private static void test4() {
        var list = Arrays.asList(1, 2, 3, 4, 5);
        var result = list.stream()
                .filter(n -> n > 3)
                .findFirst();
        System.out.println(result);
    }

    /*
    5. 避免副作用
    Stream API 旨在对数据执行功能操作。避免引入副作用，例如修改流外部的变量或执行 I/O 操作，因为这可能会导致不可预测的行为并降低代码可读性。
    */
    private static void test5() {
        var list = Arrays.asList("apple", "banana", "cherry");
        AtomicInteger count = new AtomicInteger(0);
        list.stream()
                .filter(s -> s.startsWith("a"))
                .forEach(s -> count.incrementAndGet());
        System.out.println(list);
    }

    /*
    6. 将流与不可变对象一起使用
    Stream API 最适合不可变对象。使用不可变对象可确保流的状态在处理过程中不会被修改，这可以带来更可预测的行为和更好的代码可读性
    */
    private static void test6() {
        var list = Arrays.asList("apple", "banana", "cherry");
        var result = list.stream()
                .map(String::toUpperCase)
                .collect(Collectors.toList());
        System.out.println(result);
    }

    /*
    7.在map()之前使用filter()以避免不必要的处理
    如果你的流可能包含大量不符合你的条件的元素，
    请在 map() 之前使用 filter()以避免不必要的处理。这可以提高代码的性能。
    */
    private static void test7() {

        var list = Arrays.asList(1, 2, 3, 4, 5);
        var filteredList = list.stream()
                .filter(i -> i % 2 == 0)
                .map(i -> i * 2)
                .collect(Collectors.toList());
        System.out.println(filteredList);
    }

    /*
    8.优先选择方法引用而不是 lambda 表达式
    与使用 lambda 表达式相比，方法引用可以使我们的代码更加简洁和可读。在合适的情况下，优先使用方法引用代替 lambda 表达式。
    */
    private static void test8() {
        var list = Arrays.asList(1, 2, 3, 4, 5);
        var sum = list.stream()
                .reduce(0, Integer::sum);
        System.out.println(sum);
    }

    /*
    9. 使用distinct()删除重复项
    如果你的流可能包含重复元素，请使用distinct() 操作来删除它们
    */
    private static void test9() {
        var list = Arrays.asList(1, 2, 3, 3, 4, 5, 5);
        var distinctList = list.stream()
                .distinct()
                .collect(Collectors.toList());
        System.out.println(distinctList);
    }

    /*
    10. 谨慎使用sorted()
    Sorted() 操作可能会很昂贵，尤其是对于大型流。仅在必要时谨慎使用。如果你确定输入的数据已经排序，则可以跳过此操作。
    */
    private static void test10() {
        var list = Arrays.asList(3, 2, 1);
        var sortedList = list.stream()
                .sorted()
                .collect(Collectors.toList());
        System.out.println(sortedList);
    }


    /*
    总之，Java Stream API 是一个强大而灵活的工具，可以显著简化数据处理任务的代码。
    通过遵循本文中讨论的提示，可以确保代码既高效又有效。但是，请务必记住，有效使用 Java Stream API 需要充分了解其功能和限制。
    不断学习和探索 Java Stream API 的世界，释放其全部潜力。
    */
    public static void main(String[] args) {
        test1();
        test2();
        test3();
        test4();
        test5();
        test6();
        test7();
        test8();
        test9();
        test10();
    }


}
