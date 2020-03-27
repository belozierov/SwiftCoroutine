//
//  CoPromiseTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.02.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoPromiseTests: XCTestCase {
    
    struct TestError: Error {}
    
    func testResult() {
        let promise = CoPromise<Int>()
        promise.success(1)
        XCTAssertEqual(promise.result, 1)
    }
    
    func testResult2() {
        let promise = CoPromise<Int>()
        promise.fail(TestError())
        do {
            _ = try promise.result?.get()
            XCTFail()
        } catch {
            XCTAssertTrue(error is TestError)
        }
    }
    
    func testResult3() {
        let promise = CoPromise<Int>()
        promise.success(1)
        XCTAssertEqual(promise.result, 1)
    }
    
    func testFutureResult() {
        let future = CoFuture(value: 1)
        let promise = CoPromise<Int>()
        promise.complete(with: future)
        XCTAssertEqual(promise.result, 1)
    }
    
    func testPromiseInit() {
        let future = CoFuture(value: 1)
        XCTAssertEqual(future.result, 1)
    }
    
    func testPromiseInit2() {
        let future = CoFuture<Int>(error: TestError())
        do {
            _ = try future.result?.get()
            XCTFail()
        } catch {
            XCTAssertTrue(error is TestError)
        }
    }
    
}
