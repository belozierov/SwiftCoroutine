//
//  CoFutureOperatorsTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 11.01.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
import SwiftCoroutine

class CoFutureOperatorsTests: XCTestCase {
    
    func testTransformResult() {
        let promise = CoPromise<Int>()
        let transform1 = promise.transform { try $0.get() + 1 }
        let transform2 = transform1.transformOutput { $0 + 2 }
        promise.send(1)
        promise.cancel()
        let transform3 = promise.transform { try $0.get() + 3 }
        let transform4 = transform3.transformOutput { $0 + 4 }
        XCTAssertEqual(promise.result, 1)
        XCTAssertEqual(transform1.result, 2)
        XCTAssertEqual(transform2.result, 4)
        XCTAssertEqual(transform3.result, 4)
        XCTAssertEqual(transform4.result, 8)
    }
    
    func testTransformCancel() {
        let promise = CoPromise<Int>()
        let transform1 = promise.transform { try $0.get() + 1 }
        let transform2 = transform1.transformOutput { $0 + 2 }
        promise.cancel()
        promise.send(1)
        let transform3 = promise.transform { try $0.get() + 3 }
        let transform4 = transform3.transformOutput { $0 + 4 }
        XCTAssertTrue(promise.isCancelled)
        XCTAssertTrue(transform1.isCancelled)
        XCTAssertTrue(transform2.isCancelled)
        XCTAssertTrue(transform3.isCancelled)
        XCTAssertTrue(transform4.isCancelled)
    }
    
    func testHandlersResult() {
        let exp = expectation(description: "testHandlersResult")
        exp.expectedFulfillmentCount = 19
        let promise = CoPromise<Int>()
        let transform = promise.transformOutput { $0 + 1 }
        let handler = transform.onCompletion(execute: { exp.fulfill() })
        func testTransform<T>(_ future: CoFuture<T>) {
            future.onResult { _ in exp.fulfill() }
            future.onSuccess { _ in exp.fulfill() }
            future.onCompletion(execute: exp.fulfill)
            future.onError { _ in XCTFail() }
            future.onFutureError(.cancelled) { XCTFail() }
        }
        [promise, transform, handler].forEach(testTransform)
        promise.send(1)
        promise.cancel()
        [promise, transform, handler].forEach(testTransform)
        wait(for: [exp], timeout: 1)
    }

    func testHandlersCancel() {
        let exp = expectation(description: "testHandlersCancel")
        exp.expectedFulfillmentCount = 31
        let promise = CoPromise<Int>()
        let transform = promise.transformOutput { $0 + 1 }
        let handler = transform.onCompletion(execute: { exp.fulfill() })
        func testTransform<T>(_ future: CoFuture<T>) {
            future.onResult { _ in exp.fulfill() }
            future.onSuccess { _ in XCTFail() }
            future.onCompletion(execute: exp.fulfill)
            future.onError { _ in exp.fulfill() }
            future.onFutureError(.cancelled, execute: { exp.fulfill() })
            future.onCancel(execute: exp.fulfill)
        }
        [promise, transform, handler].forEach(testTransform)
        promise.cancel()
        promise.send(1)
        [promise, transform, handler].forEach(testTransform)
        wait(for: [exp], timeout: 3)
    }
    
}
