//
//  CoFuturePerformanceTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 15.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

#if canImport(Combine)
import XCTest
import Combine
@testable import SwiftCoroutine

class CoFuturePerformanceTests: XCTestCase {

    func testMeasure() {
        measure {
            for _ in 0..<10_000 {
                let promise = CoPromise<Int>()
                let a = promise.map { $0 + 1 }
                a.whenComplete { _ in }
                a.whenComplete { _ in }
                a.map { $0 + 1 }.whenComplete { _ in }
                promise.success(0)
                let b = promise.map { $0 + 1 }
                b.whenComplete { _ in }
                b.whenComplete { _ in }
                b.map { $0 + 1 }.whenComplete { _ in }
            }
        }
    }
    
    func testMeasure2() {
        measure {
            for _ in 0..<10_000 {
                let promise = CoPromise<Int>()
                promise.map { $0 + 1 }
                    .always { _ in }
                    .always { _ in }
                    .map { $0 + 1 }
                    .whenComplete { _ in }
                promise.success(0)
                promise.map { $0 + 1 }
                    .always { _ in }
                    .always { _ in }
                    .map { $0 + 1 }
                    .whenComplete { _ in }
            }
        }
    }
    
    @available(OSX 10.15, *)
    func testCombine() {
        measure {
            for _ in 0..<10_000 {
                var promise: ((Swift.Result<Int, Never>) -> Void)!
                let future = Future<Int, Never> { promise = $0 }
                let a = future.map { $0 + 1 }
                let s1 = a.sink { _ in }
                let s2 = a.sink { _ in }
                let s3 = a.map { $0 + 1 }.sink { _ in }
                promise(.success(0))
                let b = future.map { $0 + 1 }
                let s4 = b.sink { _ in }
                let s5 = b.sink { _ in }
                let s6 = b.map { $0 + 1 }.sink { _ in }
            }
        }
    }

}
#endif
