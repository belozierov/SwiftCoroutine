////
////  CoFutureTests.swift
////  SwiftCoroutine
////
////  Created by Alex Belozierov on 10.01.2020.
////  Copyright © 2020 Alex Belozierov. All rights reserved.
////
//
//import XCTest
//@testable import SwiftCoroutine
//
//class CoFutureTests: XCTestCase {
//    
//    var future: CoFuture<Bool> {
//        CoFuture()
//    }
//
//    func testCancel() {
//        let future = self.future
//        XCTAssertFalse(future.isCancelled)
//        future.cancel()
//        XCTAssert(future.isCancelled)
//        if case .failure(let error as CoFutureError)? = future.result {
//            XCTAssertEqual(error, .cancelled)
//        } else {
//            XCTFail()
//        }
//    }
//
//    func testResult() {
//        let future = self.future
//        XCTAssertNil(future.result)
//        future.complete(with: .success(true))
//        future.complete(with: .success(false))
//        future.cancel()
//        XCTAssertEqual(future.result, true)
//        XCTAssertEqual(future.resultStorage, true)
//    }
//    
//    func testSubscribe() {
//        let future = self.future
//        let baseCount = future.subscriptions?.count ?? 0
//        let identifier = UUID()
//        future.subscribe(with: identifier) { _ in }
//        future.subscribe(with: identifier) { _ in }
//        XCTAssertEqual(future.subscriptions?.count, baseCount + 1)
//        XCTAssertTrue(future.subscriptions?.keys.contains(identifier))
//        XCTAssertNotNil(future.unsubscribe(identifier))
//        XCTAssertEqual(future.subscriptions?.count, baseCount)
//        future.subscribe(with: identifier) { _ in }
//        XCTAssertFalse(future.subscriptions?.isEmpty)
//        future.complete(with: .success(true))
//        XCTAssert(future.subscriptions?.isEmpty != false)
//        future.subscribe(with: identifier) { _ in }
//        XCTAssert(future.subscriptions?.isEmpty != false)
//    }
//    
//    func testSubscribe2() {
//        let future = self.future
//        let identifier = UUID()
//        let expectations = XCTOrderedExpectation(count: 3)
//        future.subscribe(with: identifier) { _ in
//            expectations.fulfill(1)
//        }
//        expectations.fulfill(0)
//        future.complete(with: .success(true))
//        future.subscribe(with: identifier) { _ in
//            expectations.fulfill(2)
//        }
//        wait(for: expectations, timeout: 1)
//        
//    }
//    
//    func testEqutable() {
//        let future = self.future
//        let future2 = self.future
//        XCTAssertNotEqual(future, future2)
//        XCTAssertEqual(future, future)
//        XCTAssertNotEqual(future.hashValue, future2.hashValue)
//        XCTAssertEqual(future.hashValue, future.hashValue)
//    }
//    
//}
