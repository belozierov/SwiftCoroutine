////
////  CoFutureAwaitTests.swift
////  SwiftCoroutine
////
////  Created by Alex Belozierov on 13.01.2020.
////  Copyright © 2020 Alex Belozierov. All rights reserved.
////
//
//import XCTest
//@testable import SwiftCoroutine
//
//class CoFutureAwaitTests: XCTestCase {
//
//    func testOutCoroutineCall() {
//        let promise = CoPromise<Int>()
//        do {
//            _ = try promise.await()
//            XCTFail()
//        } catch let error as Coroutine.CoroutineError {
//            return XCTAssertEqual(error, .mustBeCalledInsideCoroutine)
//        } catch {
//            XCTFail()
//        }
//    }
//
//    func testAwait() {
//        testAwait(sec: 1) { promise in
//            XCTAssertEqual(try promise.await(), 1)
//        }
//    }
//
//    func testAwaitResult() {
//        testAwait(sec: 1) { promise in
//            XCTAssertEqual(promise.awaitResult(), 1)
//        }
//    }
//
//    func testAwaitTimeout() {
//        testAwait(sec: 1) { promise in
//            XCTAssertEqual(try promise.await(timeout: .now() + 2), 1)
//        }
//    }
//
//    func testAwaitTimeout2() {
//        testAwait(sec: 2) { promise in
//            do {
//                _ = try promise.await(timeout: .now() + 1)
//            } catch let error as CoFutureError {
//                return XCTAssertEqual(error, .timeout)
//            }
//            XCTFail()
//        }
//    }
//
//    private func testAwait(sec: Int, test: @escaping (CoPromise<Int>) throws -> Void) {
//        let exp = expectation(description: "testAwaitTimeout")
//        let promise = CoPromise<Int>()
//        DispatchQueue.global().asyncAfter(deadline: .now() + .seconds(sec)) {
//            promise.send(1)
//        }
//        coroutine {
//            try test(promise)
//            exp.fulfill()
//        }
//        wait(for: [exp], timeout: 2)
//    }
//
//}
